"""Train-only typed callable feasibility study. Run from the repository root."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
import warnings

from callable_candidate_runner import MAX_RESULT, encode
from callable_fixture_reference import ABSTAIN, PUBLISHED, REFERENCES, max_mex_subset
from source_example_audit import CALL_CASE, HERE, SEED, SNAPSHOT, in_top50, read_json, workload, write_json
from src import execute
from src.config import EXEC_TIMEOUT
from verification_completion import run_job


SOURCES = ("codewars.com", "leetcode.com")
WORKERS = 2
HARNESS = HERE / "callable_candidate_runner.py"
LABEL_PAIR = re.compile(r"(?ms)^[ \t]*Input:[ \t]*(.*?)^[ \t]*Output:[ \t]*([^\r\n]*)")


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def statement_pairs(statement, owner, name, params):
    pairs = []
    if owner is None:
        for expression, return_text in CALL_CASE.findall(statement):
            try:
                call = ast.parse(expression, mode="eval").body
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == name and not call.keywords:
                    pairs.append((tuple(ast.literal_eval(x) for x in call.args), ast.literal_eval(return_text.strip())))
            except (SyntaxError, ValueError):
                continue
        return pairs

    for arg_text, return_text in LABEL_PAIR.findall(statement):
        try:
            data = arg_text.strip()
            if re.match(r"^\w+\s*=", data):
                data = re.sub(r"\s*\n\s*(?=\w+\s*=)", ", ", data)
                call = ast.parse(f"arguments({data})", mode="eval").body
                if call.args or {kw.arg for kw in call.keywords} != set(params) or len(call.keywords) != len(params):
                    continue
                values = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
                args = tuple(values[name] for name in params)
            elif len(params) == 1:
                args = (ast.literal_eval(data),)
            else:
                continue
            value = return_text.strip()
            expected = json.loads(value) if value in {"true", "false", "null"} else ast.literal_eval(value)
            pairs.append((args, expected))
        except (SyntaxError, ValueError):
            continue
    return pairs


def signature(starter, source):
    body = ast.parse(starter.rstrip() + "\n        pass\n" if source == SOURCES[1]
                     else starter.rstrip() + "\n    pass\n").body
    if len(body) != 1:
        raise ValueError("unsupported starter structure")
    owner = None
    entry = body[0]
    if source == SOURCES[1]:
        if not isinstance(entry, ast.ClassDef) or entry.name != "Solution" or len(entry.body) != 1:
            raise ValueError("unsupported starter class")
        owner = entry.name
        entry = entry.body[0]
    if not isinstance(entry, ast.FunctionDef) or entry.decorator_list:
        raise ValueError("unsupported starter callable")
    args = entry.args
    if args.kwonlyargs or args.vararg or args.kwarg or args.defaults or args.posonlyargs:
        raise ValueError("unsupported starter parameters")
    names = [arg.arg for arg in args.args]
    if owner and (not names or names.pop(0) != "self"):
        raise ValueError("unsupported receiver")
    return owner, entry.name, names


def fixture_plan(queries, metadata, audit, query_ids):
    reviewed, pools, fixtures = {}, {}, {}
    for source in SOURCES:
        pool = [qid for qid in query_ids if audit[qid]["source"] == source and
                (audit[qid]["literal_callable_example"] if source == SOURCES[0] else audit[qid]["format"] == "colon_labelled")]
        initial = random.Random(f"{SEED}:{source}").sample(pool, 12)
        opportunity = [qid for qid in pool if qid not in initial and
                       any(rank is not None and 2 <= rank <= 50 for rank in
                           (audit[qid]["rank"], audit[qid]["training_corpus_rank"]))]
        enriched = random.Random(f"{SEED}:rank:{source}").sample(opportunity, 6)
        pools[source] = {"shape_pool": pool, "uniform_review": initial, "rank_enriched_review": enriched,
                         "rank_enrichment_pool": opportunity}
        for qid in initial + enriched:
            if qid in ABSTAIN:
                reviewed[qid] = {"source": source, "abstain": ABSTAIN[qid]}
                continue
            starter = metadata[qid]["starter_code"]
            owner, name, params = signature(starter, source)
            supplied = statement_pairs(queries[qid], owner, name, params)
            cases = []
            for args, expected in PUBLISHED[qid]:
                encoded = [encode(args), encode(expected)]
                assert len(args) == len(params), qid
                assert encoded in [[encode(a), encode(r)] for a, r in supplied], (qid, args, expected)
                independent = REFERENCES[qid](*args)
                assert encode(independent) == encoded[1], (qid, args, expected, independent)
                cases.append({"args": encoded[0], "expected": encoded[1]})
            assert len({json.dumps(case["args"]) for case in cases}) == len(cases) > 0
            reviewed[qid] = {"source": source, "published_cases": len(cases)}
            fixtures[qid] = {"owner": owner, "name": name, "params": params, "cases": cases,
                             "url": metadata[qid]["url"], "statement_sha256": sha(queries[qid]),
                             "starter_sha256": sha(starter), "starter": starter}
    assert set(PUBLISHED) == set(REFERENCES) == set(fixtures)
    assert set(ABSTAIN) == reviewed.keys() - fixtures.keys()
    return {"revision": SNAPSHOT.name, "seed": SEED, "pools": pools, "reviewed": reviewed, "fixtures": fixtures}


@lru_cache(maxsize=6000)
def inspect_source(code):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            nodes = ast.parse(code).body
    except (SyntaxError, ValueError, RecursionError):
        return "invalid_python", [], []
    functions = [node for node in nodes if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [node for node in nodes if isinstance(node, ast.ClassDef)]
    classes = [node for node in classes if any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                                               for item in node.body)]
    if any(node.name == "Solution" for node in classes):
        return "solution_method", functions, classes
    if functions and classes:
        return "function_and_other_class", functions, classes
    if classes:
        return "other_class_method", functions, classes
    return ("standalone_function" if functions else "no_direct_callable"), functions, classes


def supports_call(node, nargs, receiver):
    if not isinstance(node, ast.FunctionDef) or node.decorator_list:
        return False
    args = node.args
    positional = len(args.args) + len(args.posonlyargs)
    bound = nargs + receiver
    return (not args.vararg and not args.kwarg and not args.kwonlyargs and
            positional - len(args.defaults) <= bound <= positional and positional >= receiver)


def compatibility(code, fixture):
    structure, functions, classes = inspect_source(code)
    if structure in {"invalid_python", "no_direct_callable"}:
        return structure
    owner, name = fixture["owner"], fixture["name"]
    nodes = functions
    if owner:
        owners = [node for node in classes if node.name == owner]
        if len(owners) != 1:
            return "missing_or_ambiguous_owner"
        cls = owners[0]
        if cls.bases or cls.keywords or cls.decorator_list:
            return "unsupported_owner"
        nodes = [node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        constructors = [node for node in nodes if node.name in {"__new__", "__init__"}]
        if len(constructors) > 1 or any(not supports_call(node, 0, 1) for node in constructors):
            return "unsupported_constructor"
    matches = [node for node in nodes if node.name == name]
    if len(matches) != 1:
        return "missing_or_ambiguous_name"
    return "compatible" if supports_call(matches[0], len(fixture["params"]), bool(owner)) else "unsupported_signature"


def run_case(code, fixture, case, timeout=EXEC_TIMEOUT):
    if os.name != "nt":
        raise RuntimeError("Study execution requires the Windows Job Object launcher")
    start = time.perf_counter()
    record = {"status": "runner_error"}
    try:
        with tempfile.TemporaryDirectory(prefix="callable_") as directory:
            root = Path(directory)
            request = {"owner": fixture["owner"], "name": fixture["name"], "args": case["args"]}
            (root / "solution.py").write_text(code, encoding="utf-8")
            (root / "request.json").write_text(json.dumps(request, ensure_ascii=True), encoding="ascii")
            process = execute._run_candidate([sys.executable, "-I", "-S", "-B", str(HARNESS)], input="",
                                             timeout=timeout, cwd=directory)
            record.update(returncode=process.returncode, stdout=(process.stdout or "")[:360],
                          stderr=(process.stderr or "")[-800:])
            if process.returncode:
                record["status"] = "nonzero"
            else:
                path = root / "return.json"
                if not path.is_file() or path.stat().st_size > MAX_RESULT:
                    record["status"] = "invalid_return_record"
                else:
                    value = read_json(path)
                    if value == {"unsupported_value": True}:
                        record["status"] = "unsupported_return"
                    elif isinstance(value, dict) and set(value) == {"value"}:
                        record["return"] = value["value"]
                        # JSON comparison preserves bool vs int even for a malformed record.
                        record["status"] = "pass" if json.dumps(value["value"]) == json.dumps(case["expected"]) else "mismatch"
                    else:
                        record["status"] = "invalid_return_record"
    except subprocess.TimeoutExpired:
        record["status"] = "timeout"
    except (OSError, ValueError, RuntimeError) as err:
        record.update(status="runner_error", error=repr(err))
    record["seconds"] = time.perf_counter() - start
    return record


def candidate_trial(code, fixture):
    return [run_case(code, fixture, case) for case in fixture["cases"]]


def run_controls(queries, docs, gold):
    values, mex = [1, 2, 3], 3
    correct = max_mex_subset(values, mex)
    assert correct == 2
    independent = run_job("q541", [gold["q541"], "d1030"], 2, queries, docs, gold,
                          ("1\n3 3\n1 2 3", str(correct)))
    return {"q541": {"expected": correct, "fixture": [values, mex], "observation": independent}}


def compute_rank(ids, gold, passes):
    if gold not in ids:
        return None
    ordered = [did for did in ids if did in passes] + [did for did in ids if did not in passes]
    return ordered.index(gold) + 1


def metric(rank):
    if rank is None or rank > 10:
        return 0.0, 0.0
    return 1 / math.log2(rank + 1), 1 / rank


def ranking_summary(rows, qids, list_kind, policy):
    pairs = [(rows[qid]["candidate_lists"][list_kind]["baseline_rank"], rows[qid]["candidate_lists"][list_kind][policy])
             for qid in qids]
    changes, entrants, exits = Counter(), [], []
    for qid, (old, new) in zip(qids, pairs):
        if old == new:
            changes["unchanged"] += 1
        elif new < old:
            changes["improved"] += 1
        else:
            changes["regressed"] += 1
        if new is not None and new <= 10 and (old is None or old > 10):
            entrants.append(qid)
        if old is not None and old <= 10 and (new is None or new > 10):
            exits.append(qid)
    means = [[100 * statistics.mean(metric(pair[i])[j] for pair in pairs) for j in range(2)] for i in range(2)]
    return {"queries": len(pairs), "baseline": dict(zip(("ndcg_at_10", "mrr_at_10"), means[0])),
            "experiment": dict(zip(("ndcg_at_10", "mrr_at_10"), means[1])), "changes": dict(changes),
            "entered_top10": entrants, "left_top10": exits,
            "improved_ids": [qid for qid, (old, new) in zip(qids, pairs) if old != new and new < old],
            "regressed_ids": [qid for qid, (old, new) in zip(qids, pairs) if old != new and new > old]}


def latency(seconds):
    return {"total": sum(seconds), "mean": statistics.mean(seconds), "median": statistics.median(seconds),
            "p95": sorted(seconds)[math.ceil(0.95 * len(seconds)) - 1], "count": len(seconds)}


def run_study(queries, docs, gold, fixed, ablation, qids, audit, plan):
    records = {}
    primary = "train_corpus"
    secondary = "saved_diagnostic"
    elapsed_baseline = []
    start = time.perf_counter()
    for qid in qids:
        source = audit[qid]["source"]
        candidate_lists = {primary: ablation["rows"][qid]["top"], secondary: fixed["rows"][qid]["stage1_top50"]}
        baseline_start = time.perf_counter()
        example = execute.parse_example(queries[qid])
        elapsed_baseline.append(time.perf_counter() - baseline_start)
        assert example is None and fixed["rows"][qid]["example_found"] is False
        full = fixed["rows"][qid]
        stage1 = full["stage1_gold_rank"]
        assert not full["passer_ids"] and (stage1 if in_top50(stage1) else None) == full["final_gold_rank"]
        assert all(len(ids) == len(set(ids)) == 50 for ids in candidate_lists.values())
        assert all(did in docs for did in candidate_lists[primary])
        fixture = plan["fixtures"].get(qid)
        row_start = time.perf_counter()
        union = list(dict.fromkeys(did for ids in candidate_lists.values() for did in ids if did in docs))
        candidates = {did: inspect_source(docs[did])[0] for did in union}
        statuses, tried = {}, {}
        if fixture:
            # Evaluate an absent gold separately for feasibility; it never enters either ranking.
            statuses = {did: compatibility(docs[did], fixture) for did in dict.fromkeys([*union, gold[qid]])}
            eligible = [did for did, state in statuses.items() if state == "compatible"]
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                jobs = pool.map(lambda did: candidate_trial(docs[did], fixture), eligible)
                tried = dict(zip(eligible, jobs))
        first = {did for did, results in tried.items() if results[0]["status"] == "pass"}
        passed = {did for did, results in tried.items() if all(case["status"] == "pass" for case in results)}
        repeated = passed if fixture and len(fixture["cases"]) >= 2 else set()
        row_elapsed = time.perf_counter() - row_start
        ranked = {}
        for kind, ids in candidate_lists.items():
            old = compute_rank(ids, gold[qid], set())
            saved = ablation["rows"][qid]["rank"] if kind == primary else full["final_gold_rank"]
            assert old == (saved if in_top50(saved) else None)
            ranked[kind] = {"ids": ids, "baseline_rank": old, "one_case_rank": compute_rank(ids, gold[qid], first),
                            "all_cases_rank": compute_rank(ids, gold[qid], passed),
                            "multiple_cases_rank": compute_rank(ids, gold[qid], repeated),
                            "nontarget_partition_count": sum(did not in docs for did in ids)}
        records[qid] = {"source": source, "gold": gold[qid], "candidate_lists": ranked, "structures": candidates,
                        "eligibility": statuses, "executions": tried, "one_case_passers": sorted(first),
                        "all_case_passers": sorted(passed), "multiple_case_passers": sorted(repeated),
                        "elapsed_seconds": row_elapsed}
        if tried:
            print(qid, "candidate calls", len(tried), "all-case passes", sorted(passed), flush=True)
    wall = time.perf_counter() - start
    populations = {"all_source_queries": qids, "reviewed": list(plan["reviewed"]), "fixture_eligible": list(plan["fixtures"]),
                   **{source: [qid for qid in qids if records[qid]["source"] == source] for source in SOURCES}}
    ranking = {kind: {name: {policy: ranking_summary(records, ids, kind, policy) for policy in
                             ("all_cases_rank", "one_case_rank", "multiple_cases_rank")}
                      for name, ids in populations.items()} for kind in (primary, secondary)}
    executions = [case for row in records.values() for entry in row["executions"].values() for case in entry]
    return {"rows": records, "ranking": ranking, "populations": populations, "timing": {
        "wall_seconds": wall, "baseline_parser": latency(elapsed_baseline),
        "fixture_query": latency([records[qid]["elapsed_seconds"] for qid in plan["fixtures"]]),
        "subprocess_attempt": latency([case["seconds"] for case in executions]),
        "statuses": dict(Counter(case["status"] for case in executions))}}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("fixtures", "run"))
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    queries, docs, gold, fixed, ablation, query_ids, metadata = workload()
    artifact = read_json(HERE / "source_example_audit_train.json")
    audit = artifact["rows"]
    assert list(audit) == query_ids
    assert artifact["query_digest"] == sha("\n".join(query_ids))
    plan = fixture_plan(queries, metadata, audit, query_ids)
    if args.mode == "fixtures":
        write_json(args.output, plan)
        return
    saved = HERE / "callable_verified_fixtures_train.json"
    assert plan == read_json(saved)
    qids = [qid for qid in query_ids if audit[qid]["source"] in SOURCES]
    result = run_study(queries, docs, gold, fixed, ablation, qids, audit, plan)
    result["config"] = {"revision": SNAPSHOT.name, "seed": SEED, "query_digest": artifact["query_digest"],
                        "fixture_sha256": hashlib.sha256(saved.read_bytes()).hexdigest(),
                        "harness_sha256": hashlib.sha256(HARNESS.read_bytes()).hexdigest(),
                        "windows_job_sha256": hashlib.sha256(Path(execute._windows_job.__file__).read_bytes()).hexdigest(),
                        "python": sys.version, "platform": sys.platform, "logical_processors": os.cpu_count(),
                        "timeout_seconds": EXEC_TIMEOUT, "workers": WORKERS, "no_verdict_cache": True}
    result["independent_controls"] = run_controls(queries, docs, gold)
    result["provenance"] = {"sample_size": len(audit),
                            "saved_top50": sum(in_top50(row["rank"]) for row in audit.values()),
                            "train_top50": sum(in_top50(row["training_corpus_rank"]) for row in audit.values()),
                            "both_top50": sum(in_top50(row["rank"]) and in_top50(row["training_corpus_rank"])
                                              for row in audit.values()),
                            "codeforces_raw_parses": [qid for qid, row in audit.items()
                                                      if row["source"] == "codeforces.com" and row["current"]]}
    write_json(args.output, result)


if __name__ == "__main__":
    main()
