"""Audit example formats on the saved 500 training queries without changing the verifier."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
from urllib.parse import urlsplit

import pyarrow.parquet as pq

from verification_research import COLON_EXAMPLE, SNAPSHOT, WORKERS, load_train
from verification_completion import run_job
from src import execute
from src.config import EXEC_TIMEOUT

HERE = Path(__file__).resolve().parent
SEED = 20261125
STDIO_SOURCES = {"codechef.com", "codeforces.com", "atcoder.jp", "hackerrank.com"}
HEADING = re.compile(r"(?m)^(-{5}|={5})([^=\r\n]+?)\1[ \t]*(?=\r?$)")
COMBINED = re.compile(
    r"(?ims)^Input(?::[ \t]*|[ \t]*\r?\n)(.*?)\r?\n[ \t]*Output(?::[ \t]*|[ \t]*\r?\n)(.*)"
)
LABELS = re.compile(r"(?ims)^[ \t]*Input[ \t]*:[ \t]*(.*?)\r?\n[ \t]*Output[ \t]*:[ \t]*(.*)")
NEXT = re.compile(r"(?im)^[ \t]*(?:Input[ \t]*(?::|$)|(?:Explanation|Notes?)[ \t]*:)")
NAMED = re.compile(r"(?i)(sample|example) (input|output)(?:[ \t]+([0-9]+))?")
NUMERIC_FIRST_PARAGRAPH = re.compile(r"(?:[+\-0-9.eE \t\r\n]+|yes|no)", re.I)
NARRATIVE = re.compile(r"(?i)\b(?:examples?|sample cases?|for instance|e\.g\.)\b|(?:=>|--?>|==)[ \t]*(?![<>=])")
UNFINISHED = re.compile(r"(?im)(?:will not have|no) sample (?:input|output)|\binput is truncated\b|"
                        r"<[^>]*(?:truncat|sample)[^>]*>|^[ \t]*\.{3}[ \t]*$")
COMMENTARY = re.compile(r"(?im)^[ \t]*\*?(?:try to |please |note[ \t]*:)")
CALL_CASE = re.compile(
    r"(?m)^[ \t]*(?:>>>[ \t]*)?([A-Za-z_]\w*\([^;\r\n]*\))[ \t]*(?:-{1,2}>|=>|==)[ \t]*([^#\r\n]+)"
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    with path.open("x", encoding="utf-8") as out:
        json.dump(data, out, ensure_ascii=True, separators=(",", ":"))
        out.write("\n")


def workload():
    queries, docs, gold = load_train(SNAPSHOT)
    fixed = read_json(HERE / "retrieval_diagnostic_train.json")
    ablation = read_json(HERE / "query_example_baseline_train.json")
    qids = random.Random(SEED).sample(sorted(queries), 500)
    assert fixed["query_split"] == "train" and fixed["dataset_revision"] == SNAPSHOT.name
    assert qids == fixed["sampling"]["query_ids"] == ablation["populations"]["uniform"]
    (query_file,) = SNAPSHOT.glob("queries/*.parquet")
    metadata = pq.read_table(query_file, columns=["_id", "meta_information"],
                             filters=[("partition", "=", "train")]).to_pylist()
    return queries, docs, gold, fixed, ablation, qids, {r["_id"]: r["meta_information"] for r in metadata}


def sections(query):
    headings = list(HEADING.finditer(query))
    return [(m[2].strip().casefold().rstrip(":").rstrip(), query[m.end():end].strip(), m.start())
            for m, end in zip(headings, [m.start() for m in headings[1:]] + [len(query)])]


def usable_literals(query, starter):
    """Count obvious literal function examples separately from runnable stdin."""
    names = set(re.findall(r"(?m)^[ \t]*def ([a-zA-Z_]\w*)\(", starter or ""))
    for call, output in CALL_CASE.findall(query):
        try:
            node = ast.parse(call, mode="eval").body
            if not isinstance(node, ast.Call) or node.func.id not in names or not node.args or node.keywords:
                continue
            for arg in node.args:
                ast.literal_eval(arg)
            ast.literal_eval(output.strip())
            return True
        except (AttributeError, SyntaxError, ValueError):
            continue
    return False


def inspect(query, source, starter):
    blocks = sections(query)
    original = execute.parse_example(query)
    narrow = COLON_EXAMPLE.search(query)
    old_colon = (narrow[2].strip(), narrow[4].strip()) if not original and narrow and (narrow[1] or narrow[3]) else None
    generic = LABELS.search(query)
    plain_colon = (generic[1].strip(), generic[2].splitlines()[0].strip()) if generic and generic[2].splitlines() else None
    forms = []
    for label, text, offset in blocks:
        if label in {"example", "examples", "sample", "samples"}:
            match = COMBINED.match(text)
            if match:
                exp = NEXT.split(match[2], maxsplit=1)[0].strip()
                family = "colon_labelled" if re.search(r"(?im)^(?:input|output):", match[0]) else "explicit_pairs"
                forms.append((offset, family, match[1].strip(), exp, "combined"))
        named = NAMED.fullmatch(label)
        if named and named[2].lower() == "input":
            counterpart = next((data for heading, data, pos in blocks
                                if pos > offset and heading == f"{named[1]} output" + (f" {named[3]}" if named[3] else "")), None)
            if counterpart is None:
                continue
            family = "paired_headings"
            output = NEXT.split(counterpart, maxsplit=1)[0].strip()
            if source == "atcoder.jp":
                paragraphs = re.split(r"\r?\n[ \t]*\r?\n", output, maxsplit=1)
                if len(paragraphs) == 2 and NUMERIC_FIRST_PARAGRAPH.fullmatch(paragraphs[0]):
                    output = paragraphs[0]
                    family = "atcoder_first_paragraph"
            forms.append((offset, "explicit_pairs", text, output, family))
    forms.sort()
    pair = next((entry for entry in forms if entry[2] and entry[3]), None)
    labels = [label for label, _, _ in blocks]
    source_is_callable = source in {"codewars.com", "leetcode.com"}
    issue = None
    if pair and UNFINISHED.search(pair[2] + "\n" + pair[3]):
        issue = "placeholder_or_truncation"
    elif pair and COMMENTARY.search(pair[2] + "\n" + pair[3]):
        issue = "unseparated_commentary"
    elif source_is_callable and (pair or plain_colon):
        issue = "function_arguments_not_stdin"
    elif not pair and (forms or any(NAMED.fullmatch(label) for label in labels)
                       or any(re.search(r"example.*(?:input|output)", line, re.I)
                              for line in query.splitlines() if line.startswith("-----"))):
        issue = "incomplete_or_flattened_labels"
    if issue in {"placeholder_or_truncation", "unseparated_commentary"}:
        fmt = "ambiguous"
    elif pair:
        fmt = pair[1]
    elif plain_colon:
        fmt = "colon_labelled"
    elif issue:
        fmt = "ambiguous"
    elif NARRATIVE.search(query):
        fmt = "narrative"
    elif re.search(r"(?im)^.*(?:constraints|input format|output format).*$", query) or {"input", "output"} <= set(labels):
        fmt = "constraints_or_spec_only"
    else:
        fmt = "no_usable_example"
    proposal = list(pair[2:4]) if pair and source in STDIO_SOURCES and not issue else None
    return {"format": fmt, "issue": issue, "current": original,
            "proposed": proposal, "proposal_family": pair[4] if pair else None,
            "previous_colon_probe": old_colon, "ungated_colon_labels": plain_colon,
            "starter_present": bool(starter and starter.strip()),
            "literal_callable_example": usable_literals(query, starter) if source_is_callable else False}


def inventory():
    queries, docs, gold, fixed, ablation, qids, metadata = workload()
    rows = {}
    for qid in qids:
        meta = metadata[qid]
        source = urlsplit(meta["url"]).hostname.removeprefix("www.").lower()
        saved = fixed["rows"][qid]
        row = inspect(queries[qid], source, meta["starter_code"])
        assert saved["gold_id"] == gold[qid]
        rows[qid] = {"source": source, "gold_id": gold[qid], "rank": saved["stage1_gold_rank"],
                     "training_corpus_rank": ablation["rows"][qid]["rank"],
                     **row}
    return rows


def measure(rows):
    queries, docs, gold, fixed, _, _, _ = workload()
    doc_ids = sorted(docs)
    trials = {}
    for qid, row in rows.items():
        if not row["proposed"] or row["current"]:
            continue
        did = row["gold_id"]
        eligible = [d for d in doc_ids if d != did and docs[d] != docs[did]]
        random_controls = random.Random(f"{SEED}:{qid}").sample(eligible, 5)
        hits = row["rank"] is not None and row["rank"] <= 50
        nearest_controls = [d for d in fixed["rows"][qid]["stage1_top50"]
                            if d in docs and d != did and docs[d] != docs[did]] if hits else []
        known_controls = ["d814"] if qid == "q949" else []
        ordered = list(dict.fromkeys([did, *nearest_controls, *random_controls, *known_controls]))
        trial = run_job(qid, ordered, row["rank"], queries, docs, gold, tuple(row["proposed"]))
        trials[qid] = {"random_controls": random_controls, "near_training_controls": nearest_controls,
                       "known_wrong_controls": known_controls, **trial}
        if len(trials) % 10 == 0:
            print(f"{len(trials)} proposal trials complete", flush=True)
    return trials


def in_top50(rank):
    return rank is not None and rank <= 50


def summarize(audit, labels):
    rows, trials = audit["rows"], audit["trials"]
    assert len(rows) == 500 and labels.keys() <= rows.keys()
    assert hashlib.sha256("\n".join(rows).encode()).hexdigest() == audit["query_digest"]
    sources = {}
    for source in sorted({row["source"] for row in rows.values()}):
        ids = [qid for qid, row in rows.items() if row["source"] == source]
        retrieved = [qid for qid in ids if in_top50(rows[qid]["rank"])]
        totals = {}
        for scope, population in (("all", ids), ("saved_top50", retrieved)):
            counts = Counter()
            for qid in population:
                row = rows[qid]
                kind = labels.get(qid, {}).get("kind")
                potential = kind != "unsafe_stdio" and bool(
                    row["current"] or row["proposed"] or kind in {"manual_stdio", "literal_stdio_review_only"}
                )
                counts.update({"queries": 1, "current_raw": bool(row["current"]),
                               "potential_stdio": potential,
                               "potential_recognized": potential and bool(row["current"]),
                               "potential_missed": potential and not row["current"],
                               "literal_callable_lower_bound": row["literal_callable_example"],
                               "new_trials": qid in trials,
                               "new_gold_matches": qid in trials and row["gold_id"] in trials[qid]["base"]})
            totals[scope] = dict(counts)
        sources[source] = {**totals, "formats": dict(Counter(rows[qid]["format"] for qid in ids)),
                           "train_corpus_top50": sum(in_top50(rows[qid]["training_corpus_rank"]) for qid in ids)}

    kinds = sorted({label["kind"] for label in labels.values()})
    label_results = {}
    for method in ("current", "previous_colon_probe", "ungated_colon_labels", "proposed"):
        results = {kind: {"exact": [], "nonexact": [], "abstained": []} for kind in kinds}
        for qid, label in labels.items():
            extracted = rows[qid][method]
            verdict = "abstained"
            if extracted:
                verdict = "exact" if extracted == label.get("pair") else "nonexact"
            results[label["kind"]][verdict].append(qid)
        label_results[method] = results

    controls = {}
    for key in ("random_controls", "near_training_controls", "known_wrong_controls"):
        pairs = [(qid, did) for qid, trial in trials.items() for did in trial[key]]
        controls[key] = {"pairs": len(pairs), "matches": [(qid, did) for qid, did in pairs if did in trials[qid]["base"]]}
    upward = [qid for qid, trial in trials.items() if in_top50(rows[qid]["rank"]) and
              rows[qid]["rank"] > 1 and trial["gold"] in trial["base"]]
    return {"query_digest": audit["query_digest"], "sources": sources, "label_results": label_results,
            "controls": controls, "gold_phases": dict(Counter(t["outcomes"][t["gold"]]["phase"] for t in trials.values())),
            "distinct_candidate_executions": sum(len(t["candidates"]) for t in trials.values()),
            "upward_slots": upward, "upward_slots_without_observed_rival_match":
            [qid for qid in upward if trials[qid]["base"] == [trials[qid]["gold"]]]}


def witness_trials(rows, labels):
    queries, docs, gold, _, _, _, _ = workload()
    # Expected outputs are derived from the problem contracts and kept separate from published examples.
    witnesses = [
        ("q1465", "d2133", ("1\n2019:03:29", "2"), "March 29 and 31 are correct, April 1 breaks alternation."),
        ("q692", "d556", ("1\n3\n2 4 6", "0"), "No earlier smaller value is divisible by a later larger value."),
        ("q541", "d1030", ("1\n4 3\n1 2 3 5", "3"), "Choose 1, 2 and 5, and exclude the target MEX of 3."),
        ("q949", "d814", ("1 2", "0"), "The only values are 2 and 3, neither a perfect square."),
    ]
    recorded = {}
    for qid, did, witness, reason in witnesses:
        candidates = [gold[qid], did]
        recorded[qid] = {"published_example": rows[qid]["proposed"], "witness": witness,
                         "reason": reason, "published_run": run_job(qid, candidates, rows[qid]["rank"],
                                                                      queries, docs, gold, tuple(rows[qid]["proposed"])),
                         "witness_run": run_job(qid, candidates, rows[qid]["rank"], queries, docs, gold, witness)}
    manual = {
        qid: labels[qid]["pair"] for qid in ("q658", "q944", "q1211", "q2631", "q2650")
    }
    gold_defect = run_job("q541", [gold["q541"], "d1030"], rows["q541"]["rank"],
                          queries, docs, gold, ("1\n3 3\n1 2 3", "2"))
    return {"coincidental_matches": recorded,
            "gold_defect": {"input": "1\n3 3\n1 2 3", "run": gold_defect}, "manual_stdio_runs": {
        qid: {"example": pair, "run": run_job(qid, [gold[qid]], rows[qid]["rank"], queries, docs, gold, tuple(pair))}
        for qid, pair in manual.items()}, "timeout_repeats": [
            run_job("q827", [gold["q827"]], rows["q827"]["rank"], queries, docs, gold, tuple(rows["q827"]["proposed"]))
            for _ in range(3)]}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    cli.add_argument("--execute", action="store_true", help="run new stdio pairs and training-only controls")
    cli.add_argument("--summarize", type=Path, help="aggregate a saved audit with hand-labelled cases")
    cli.add_argument("--witnesses", action="store_true", help="reproduce witness runs alongside a saved audit summary")
    args = cli.parse_args()
    if args.witnesses and not args.summarize:
        cli.error("--witnesses requires --summarize")
    if args.summarize:
        audit = read_json(args.summarize)
        labels = read_json(HERE / "source_example_labels_train.json")
        result = summarize(audit, labels)
        if args.witnesses:
            result["witness_trials"] = witness_trials(audit["rows"], labels)
    else:
        rows = inventory()
        result = {"revision": SNAPSHOT.name, "seed": SEED,
                  "query_digest": hashlib.sha256("\n".join(rows).encode()).hexdigest(),
                  "run_settings": {"split": "train", "timeout_seconds": EXEC_TIMEOUT,
                                   "workers": WORKERS, "cached_verdicts": False},
                  "rows": rows, "trials": measure(rows) if args.execute else {}}
    write_json(args.output, result)
