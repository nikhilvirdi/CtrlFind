"""Audit the retained example parser using only the pinned query parquet."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import re

import pyarrow
import pyarrow.parquet as pq

from verification_research import COLON_EXAMPLE, SNAPSHOT
from src import execute
from src.config import DATASET

SEED = 20261125
SAMPLE_SIZE = 500
LINE = re.I | re.M
PATTERNS = {
    "input_colon": re.compile(r"\binput[ \t]*:", re.I),
    "output_colon": re.compile(r"\boutput[ \t]*:", re.I),
    "example_word": re.compile(r"\bexamples?\b", re.I),
    "sample_word": re.compile(r"\bsamples?\b", re.I),
    "dashed_example_heading": re.compile(r"^[ \t]*-{5}[ \t]*examples?[ \t]*:?[ \t]*-{5}", LINE),
    "numbered_or_plain_example_heading": re.compile(
        r"^[ \t]*(?:#+[ \t]*)?\*{0,2}(?:examples?|samples?)(?:[ \t]+\d+)?(?:[ \t]*:|\*{0,2}[ \t]*$)", LINE),
    "narrative_cue": re.compile(r"\b(?:for example|for instance|as an example|to illustrate)\b|\be\.g\.", re.I),
    "arrow_or_function_result_cue": re.compile(r"=>|->|→|⇒|\)[ \t]*(?:==|#[ \t]*returns?\b|[ \t]+returns?\b)", re.I),
}
SPLIT_INPUT = re.compile(r"^[ \t]*-{5}[ \t]*(?:sample|example)[ \t]+input[ \t]*:?[ \t]*-{5}", LINE)
SPLIT_OUTPUT = re.compile(r"^[ \t]*-{5}[ \t]*(?:sample|example)[ \t]+output[ \t]*:?[ \t]*-{5}", LINE)
PLAIN_INPUT = re.compile(r"^[ \t]*(?:#+[ \t]*)?\*{0,2}input[ \t]*:", LINE)
PLAIN_OUTPUT = re.compile(r"^[ \t]*(?:#+[ \t]*)?\*{0,2}output[ \t]*:", LINE)
INLINE_DASHED_LABEL = re.compile(r"^[ \t]*(?:input|output)[ \t]*:[ \t]*[^\n\s]", LINE)
FAMILIES = (
    ("retained_bare_io", "retained_regex_match"),
    ("dashed_colon_io", "dashed_standalone_colon_pair"),
    ("split_dashed_io", "split_dashed_io_headings"),
    ("other_dashed_example", "dashed_example_heading"),
    ("plain_colon_io", "plain_colon_io_lines"),
    ("plain_example_heading", "numbered_or_plain_example_heading"),
    ("narrative_cue", "narrative_cue"),
    ("arrow_or_function_result", "arrow_or_function_result_cue"),
)
NO_MARKER = "no_recognized_marker"


def classify(text):
    flags = {name: bool(pattern.search(text)) for name, pattern in PATTERNS.items()}
    colon = COLON_EXAMPLE.search(text)
    dashed = PATTERNS["dashed_example_heading"].search(text)
    dashed_body = text[dashed.end():].split("\n-----", 1)[0] if dashed else ""
    split = SPLIT_INPUT.search(text)
    plain = PLAIN_INPUT.search(text)
    flags.update(
        retained_regex_match=bool(execute.EXAMPLE_RE.search(text)),
        both_colon_markers=flags["input_colon"] and flags["output_colon"],
        dashed_standalone_colon_pair=bool(colon and (colon[1] or colon[3]) and colon[2].strip() and colon[4].strip()),
        dashed_inline_colon_payload=bool(INLINE_DASHED_LABEL.search(dashed_body)),
        split_dashed_io_headings=bool(split and SPLIT_OUTPUT.search(text, split.end())),
        plain_colon_io_lines=bool(plain and PLAIN_OUTPUT.search(text, plain.end())),
    )
    flags[NO_MARKER] = not any(flags.values())
    family = next((label for label, key in FAMILIES if flags[key]),
                  NO_MARKER if flags[NO_MARKER] else "other_marker")
    return {"parsed": execute.parse_example(text) is not None,
            "family": family, "markers": [name for name, present in flags.items() if present]}


def counts(rows, population):
    n = len(rows)
    parsed = sum(row["parsed"] for row in rows)
    return {"total": n, "parsed": parsed, "unparsed": n - parsed,
            "share_of_population_percent": 100 * n / population,
            "parsed_percent": 100 * parsed / n if n else None,
            "unparsed_percent": 100 * (n - parsed) / n if n else None}


def summarize(rows):
    n = len(rows)
    marker_names = [NO_MARKER, *PATTERNS,
        "retained_regex_match", "both_colon_markers", "dashed_standalone_colon_pair",
        "dashed_inline_colon_payload", "split_dashed_io_headings", "plain_colon_io_lines",
    ]
    result = counts(rows, n)
    result["markers_overlapping"] = {
        marker: counts([row for row in rows if marker in row["markers"]], n) for marker in marker_names
    }
    result["primary_families_exclusive"] = {
        family: counts([row for row in rows if row["family"] == family], n)
        for family in [label for label, _ in FAMILIES] + ["other_marker", NO_MARKER]
    }
    families = result["primary_families_exclusive"].values()
    assert sum(group["total"] for group in families) == n
    assert sum(group["parsed"] for group in families) == result["parsed"]
    return result


def self_check():
    cases = (
        ("-----Example-----\nInput\nx\nOutput\ny", True, "retained_bare_io"),
        ("-----Example-----\nInput:\nx\nOutput:\ny", False, "dashed_colon_io"),
        ("-----Example-----\nInput:x\nOutput:y", False, "other_dashed_example"),
        ("-----Sample Input-----\nx\n-----Sample Output-----\ny", False, "split_dashed_io"),
        ("Example 1:\nInput: x\nOutput: y", False, "plain_colon_io"),
        ("For example, doubling two produces four.", False, "narrative_cue"),
        ("Convert a string to uppercase.", False, NO_MARKER),
    )
    for text, parsed, family in cases:
        row = classify(text)
        assert (row["parsed"], row["family"]) == (parsed, family)
    assert "dashed_inline_colon_payload" in classify(cases[2][0])["markers"]


def audit(output):
    self_check()
    (path,) = SNAPSHOT.glob("queries/*.parquet")
    train = {row["_id"]: row["text"] for row in pq.read_table(
        path, columns=["_id", "text"], filters=[("partition", "=", "train")]).to_pylist()}
    qids = random.Random(SEED).sample(sorted(train), SAMPLE_SIZE)
    train_rows = {qid: classify(train[qid]) for qid in qids}
    test_texts = pq.read_table(path, columns=["text"], filters=[("partition", "=", "test")])["text"]
    test_summary = summarize([classify(text.as_py()) for text in test_texts])
    assert len(train) == 5000 and test_summary["total"] == 3765

    cases_path = Path(__file__).with_name("example_parser_audit_cases.json")
    reviewed = json.loads(cases_path.read_text(encoding="utf-8"))
    assert len(reviewed) == len({case["query_id"] for case in reviewed}) == 20
    for case in reviewed:
        qid = case["query_id"]
        assert not train_rows[qid]["parsed"]
        assert all(excerpt in train[qid] for excerpt in case["excerpts"]), qid
    result = {
        "dataset": DATASET, "dataset_revision": SNAPSHOT.name, "parquet_file": path.name,
        "parser_sha256": hashlib.sha256(Path(execute.__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(), "pyarrow": pyarrow.__version__,
        "sampling": {"seed": SEED, "training_universe": len(train),
                     "method": "random.Random(seed).sample(sorted(training_query_ids), 500)",
                     "ordered_ids_sha256": hashlib.sha256("\n".join(qids).encode()).hexdigest(), "query_ids": qids},
        "train": summarize(list(train_rows.values())), "test": test_summary,
        "reviewed_train_case_ids": [case["query_id"] for case in reviewed],
        "train_rows": train_rows,
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    for split in ("train", "test"):
        summary = result[split]
        print(f"{split}: {summary['parsed']}/{summary['total']} parsed ({summary['parsed_percent']:.2f}%)")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    audit(cli.parse_args().output)
