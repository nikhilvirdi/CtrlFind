"""Reproduce the training-only verification measurements from the research log."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import execute
from src.config import EXEC_TIMEOUT

REVISION = "f22508f96b7a36c2415181ed8bb76f76e04ae2d5"
SEED = 20261124
WRONG_PER_QUERY = 5
WORKERS = 2
COLON_PROBE_COUNT = 50
COLON_EXAMPLE = re.compile(
    r"^-{5}Examples?-{5}[ \t]*\r?\n"
    r"[ \t]*Input[ \t]*(:?)[ \t]*\r?\n(.*?)\r?\n"
    r"[ \t]*Output[ \t]*(:?)[ \t]*\r?\n(.*?)"
    r"(?=\r?\n[ \t]*Input[ \t]*:?[ \t]*\r?\n|\r?\n-----|\Z)",
    re.M | re.S,
)
SNAPSHOT = (
    Path.home() / ".cache/huggingface/hub/datasets--CoIR-Retrieval--apps/snapshots"
    / REVISION
)


def load_train(snapshot):
    import pyarrow.parquet as pq

    def read(pattern, columns, filters=None):
        (path,) = snapshot.glob(pattern)
        return pq.read_table(path, columns=columns, filters=filters).to_pylist()

    train = [("partition", "=", "train")]
    queries = {
        r["_id"]: r["text"] for r in read("queries/*.parquet", ["_id", "text"], train)
    }
    docs = {
        r["_id"]: r["text"] for r in read("corpus/*.parquet", ["_id", "text"], train)
    }
    qrels = read("data/train-*.parquet", ["query-id", "corpus-id", "score"])
    gold = {r["query-id"]: r["corpus-id"] for r in qrels if r["score"] > 0}
    assert len(qrels) == len(gold) == len(queries) == len(docs) == 5000
    assert gold.keys() == queries.keys() and set(gold.values()) == docs.keys()
    return queries, docs, gold


def sample(queries, docs, gold):
    query_rng = random.Random(SEED)
    wrong_rng = random.Random(SEED + 1)
    doc_ids = sorted(docs)
    order = sorted(queries)
    query_rng.shuffle(order)
    for qid in order:
        did = gold[qid]
        example = execute.parse_example(queries[qid])
        wrong = []
        if example is not None:
            eligible = [d for d in doc_ids if d != did and docs[d] != docs[did]]
            wrong = wrong_rng.sample(eligible, WRONG_PER_QUERY)
        yield qid, did, example, wrong


def script_fingerprint(code):
    written_bytes = code.replace("\n", os.linesep).encode("utf-8")
    return hashlib.sha256(written_bytes).hexdigest()


def observed_verify(codes, example, timeout=EXEC_TIMEOUT, workers=WORKERS):
    """Run the production verifier and record the subprocess data it discards."""
    observations = {}
    runner = execute._run_candidate

    def run(args, **kwargs):
        key = hashlib.sha256(Path(args[1]).read_bytes()).hexdigest()
        start = time.perf_counter()
        record = {"returncode": None, "stdout": None, "stderr": None}
        try:
            result = runner(args, **kwargs)
            record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            return result
        except subprocess.TimeoutExpired as exc:
            record.update(timeout=True, stdout=decode(exc.stdout), stderr=decode(exc.stderr))
            raise
        except Exception as exc:
            record["exception"] = repr(exc)
            raise
        finally:
            record["seconds"] = round(time.perf_counter() - start, 4)
            observations[key] = record

    with patch.object(execute, "_run_candidate", run):
        passed = execute.verify(codes, example, timeout=timeout, workers=workers)
    return passed, [observations.get(script_fingerprint(code)) for code in codes]


def decode(value):
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def status(passed, record):
    if passed:
        return "pass"
    if record is None or record.get("exception"):
        return "harness_error"
    if record.get("timeout"):
        return "timeout"
    if record["returncode"] != 0:
        return "nonzero_exit"
    return "mismatch"


def colon_candidates(queries):
    candidates = {}
    for qid, query in queries.items():
        if execute.parse_example(query) is not None:
            continue
        match = COLON_EXAMPLE.search(query)
        if match and (match[1] or match[3]) and match[2].strip() and match[4].strip():
            candidates[qid] = (match[2].strip(), match[4].strip())
    return candidates


def probe_colons(snapshot, output):
    queries, docs, gold = load_train(snapshot)
    candidates = colon_candidates(queries)
    qids = random.Random(SEED + 2).sample(sorted(candidates), COLON_PROBE_COUNT)
    trials = []
    for qid in qids:
        did = gold[qid]
        example = candidates[qid]
        passed, observed = observed_verify([docs[did]], example, workers=1)
        trials.append({
            "query_id": qid, "gold_id": did, "passed": passed[0],
            "stdin": example[0], "expected_stdout": example[1],
            "observation": observed[0],
        })
    with output.open("w", encoding="utf-8") as f:
        json.dump({"eligible": len(candidates), "seed": SEED + 2, "trials": trials}, f)


def measure(snapshot, output):
    queries, docs, gold = load_train(snapshot)
    counts = Counter()
    with output.open("w", encoding="utf-8") as f:
        header = {
            "split": "train", "revision": snapshot.name,
            "seed": SEED, "query_count": len(queries), "wrong_seed": SEED + 1,
            "wrong_per_query": WRONG_PER_QUERY, "workers": WORKERS,
            "timeout_seconds": EXEC_TIMEOUT, "platform": platform.platform(),
            "python": platform.python_version(),
        }
        f.write(json.dumps(header) + "\n")
        for i, (qid, did, example, wrong) in enumerate(sample(queries, docs, gold), 1):
            row = {"query_id": qid, "gold_id": did, "example_found": example is not None}
            if example is not None:
                codes = [docs[d] for d in [did, *wrong]]
                passed, observed = observed_verify(codes, example)
                row["gold_pass"] = passed[0]
                row["gold_observation"] = observed[0]
                row["wrong"] = [
                    {
                        "doc_id": d, "status": status(ok, record),
                        **({"observation": record} if ok else {}),
                    }
                    for d, ok, record in zip(wrong, passed[1:], observed[1:])
                ]
                counts["gold_pass"] += passed[0]
                counts["gold_fail"] += not passed[0]
                counts["wrong_pass"] += sum(passed[1:])
                counts["wrong_runs"] += len(wrong)
            else:
                counts["skipped"] += 1
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 500 == 0:
                print(i, dict(counts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="destination for the raw JSONL trial log")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--colon-probe", action="store_true", help="write JSON for the colon-label diagnostic")
    args = parser.parse_args()
    if args.colon_probe:
        probe_colons(args.snapshot, args.output)
    else:
        measure(args.snapshot, args.output)
