"""Run the paired verifier audit and the training-only K=50 latency baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import locale
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import verification_completion as replay
from verification_research import SNAPSHOT, WORKERS, load_train, sample
from src import execute
from src.config import DEFAULT_K, EXEC_TIMEOUT

OLD_REVISION = "0586e97"
PERFORMANCE_QUERIES = 200


def write_json(path, evidence):
    with path.open("x", encoding="utf-8") as output:
        json.dump(evidence, output, separators=(",", ":"))
        output.write("\n")


def audit(path, queries, docs, gold, examples):
    original = ModuleType("src.original_execute")
    source = subprocess.check_output(["git", "show", f"{OLD_REVISION}:src/execute.py"])
    exec(compile(source, f"{OLD_REVISION}:src/execute.py", "exec"), original.__dict__)
    assert all(original.parse_example(text) == examples.get(qid) for qid, text in queries.items())
    jobs = [(qid, [did, *wrong]) for qid, did, ex, wrong in sample(queries, docs, gold) if ex]
    saved = json.loads(Path(__file__).with_name("train_verification_audit.json").read_text(encoding="utf-8"))
    assert jobs == [(row["query_id"], [row["gold_id"], *(w["doc_id"] for w in row["wrong"])])
                    for row in saved["trial_ledger"]]

    rows = {}
    for count, (qid, ids) in enumerate(jobs, 1):
        codes = [replay.CandidateSource(docs[did]) for did in ids]
        _, records = replay.observe(codes, examples[qid])
        legacy = {}
        scoped = {}
        phases = []
        for code, obs in zip(codes, records):
            status = replay.phase(obs)
            stdout = obs.get("stdout")
            legacy[id(code)] = stdout.strip() if status in ("complete", "nonzero") and isinstance(stdout, str) else None
            scoped[id(code)] = legacy[id(code)] if status == "complete" else None
            phases.append(status)
        with patch.object(original, "run_snippet", side_effect=lambda code, *args: legacy[id(code)]):
            before = original.verify(codes, examples[qid], timeout=EXEC_TIMEOUT, workers=WORKERS)
        with patch.object(execute, "run_snippet", side_effect=lambda code, *args: scoped[id(code)]):
            after = execute.verify(codes, examples[qid], timeout=EXEC_TIMEOUT, workers=WORKERS,
                                   allow_trailing_padding=execute.allows_trailing_padding(queries[qid]))
        changes = {did: {"stdout": records[i].get("stdout"), "stderr": records[i].get("stderr"),
                         "returncode": records[i].get("returncode"), "expected": examples[qid][1]}
                   for i, did in enumerate(ids) if before[i] != after[i]}
        rows[qid] = dict(gold=ids[0], candidates=ids, phase=phases, original=before, current=after, changes=changes)
        if count % 100 == 0:
            print("audit", count, flush=True)
    for rule in ("original", "current"):
        print(rule, sum(row[rule][0] for row in rows.values()),
              sum(sum(row[rule][1:]) for row in rows.values()), flush=True)
    write_json(path, dict(revision=SNAPSHOT.name, original_commit=OLD_REVISION, workers=WORKERS,
                          timeout=EXEC_TIMEOUT, rows=rows))


class TimedHits(dict):
    def __init__(self, hits):
        super().__init__({qid: scores.copy() for qid, scores in hits.items()})
        self.seconds = {}

    def items(self):
        for qid, scores in super().items():
            start = perf_counter()
            yield qid, scores
            self.seconds[qid] = perf_counter() - start


def latency_summary(values):
    return {"median": statistics.median(values), "p95": sorted(values)[math.ceil(.95 * len(values)) - 1],
            "mean": statistics.mean(values), "sum": sum(values)}


def cpu_name():
    if sys.platform != "win32":
        return platform.processor()
    import winreg

    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()


def performance(path, plan_path, repeats, queries, docs, gold, examples):
    from src.mteb_wrappers import LiveExecReranker, SearchEncoderWrapper

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert list(plan) == random.Random(replay.RANK_SEED).sample(sorted(examples), PERFORMANCE_QUERIES)
    assert all(qid in examples and len(entry["top"]) == DEFAULT_K and set(entry["top"]) <= docs.keys()
               for qid, entry in plan.items())
    # Stage 1 order is fixed before timing. Positive descending scores preserve that order.
    hits = {qid: {did: 1 - i / 100 for i, did in enumerate(entry["top"])} for qid, entry in plan.items()}
    batch = {"id": list(plan), "text": [queries[qid] for qid in plan]}
    evidence = {
        "revision": SNAPSHOT.name, "seed": replay.RANK_SEED, "k": DEFAULT_K,
        "machine": {"cpu": cpu_name(), "os": platform.platform(), "python": sys.version,
                    "cpu_count": os.cpu_count(), "process_cpu_count": os.process_cpu_count(),
                    "utf8_mode": sys.flags.utf8_mode, "encoding": locale.getpreferredencoding(False)},
        "timeout": EXEC_TIMEOUT, "workers": os.cpu_count() or 2,
        "cache": "empty temporary live JSONL per cold run, same checkpoint reopened for warm run",
        "verifier_sha256": hashlib.sha256(Path(execute.__file__).read_bytes()).hexdigest(),
        "plan": {qid: {**entry, "gold": gold[qid]} for qid, entry in plan.items()}, "runs": [],
    }
    with tempfile.TemporaryDirectory(prefix="verifier_perf_") as scratch:
        for repeat in range(repeats):
            checkpoint = Path(scratch) / f"{repeat}.jsonl"
            live = LiveExecReranker(SimpleNamespace(mteb_model_meta=None), checkpoint, k=DEFAULT_K)
            live.doc_text = docs
            cold_results = None
            for mode in ("cold", "warm"):
                timed = TimedHits(hits)
                launches = {key: [] for key in ("timeout", "launch_error", "other_error", "exit_zero", "nonzero")}
                verdicts = {}
                qids = iter(plan)
                runner = execute._run_candidate
                verify = execute.verify

                def run(*args, **kwargs):
                    # Append per future so concurrent timeouts count individually.
                    try:
                        result = runner(*args, **kwargs)
                    except subprocess.TimeoutExpired:
                        launches["timeout"].append(None)
                        raise
                    except OSError:
                        launches["launch_error"].append(None)
                        raise
                    except Exception:
                        launches["other_error"].append(None)
                        raise
                    else:
                        launches["exit_zero" if result.returncode == 0 else "nonzero"].append(None)
                        return result

                def capture(codes, example, timeout, *, workers, allow_trailing_padding):
                    qid = next(qids)
                    assert example == examples[qid]
                    ids = plan[qid]["top"]
                    result = verify(codes, example, timeout, workers=workers,
                                    allow_trailing_padding=allow_trailing_padding)
                    verdicts[qid] = [did for did, ok in zip(ids, result) if ok]
                    return result

                with (
                    patch.object(SearchEncoderWrapper, "search", return_value=timed),
                    patch("src.mteb_wrappers.verify", side_effect=capture),
                    patch.object(execute, "_run_candidate", run),
                ):
                    start = perf_counter()
                    results = live.search(batch)
                    elapsed = perf_counter() - start
                counts = {key: len(items) for key, items in launches.items()}
                assert len(timed.seconds) == PERFORMANCE_QUERIES
                if mode == "cold":
                    assert len(verdicts) == PERFORMANCE_QUERIES
                    cold_results = {qid: results[qid].copy() for qid in plan}
                else:
                    assert results == cold_results and not verdicts and not sum(counts.values())
                run_data = dict(repeat=repeat + 1, mode=mode, wall_seconds=elapsed, query_seconds=timed.seconds,
                                latency_seconds=latency_summary(list(timed.seconds.values())), process_counts=counts,
                                passer_ids=verdicts)
                evidence["runs"].append(run_data)
                print(repeat + 1, mode, elapsed, run_data["latency_seconds"], counts, flush=True)
            # Preserve completed pairs if a subsequent repeat is interrupted.
            write_json(path.with_name(f"{path.stem}.repeat{repeat + 1}.tmp.json"), evidence)
    write_json(path, evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "plan", "rank", "performance"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.mode in ("rank", "performance") and args.plan is None:
        parser.error("rank and performance require --plan")
    if args.repeats < 1:
        parser.error("use at least one repeat")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    queries, docs, gold = load_train(SNAPSHOT)
    examples = {qid: ex for qid, query in queries.items() if (ex := execute.parse_example(query)) is not None}
    if args.mode == "audit":
        audit(args.output, queries, docs, gold, examples)
    elif args.mode == "plan":
        with patch.object(replay, "RANK_COUNT", PERFORMANCE_QUERIES):
            write_json(args.output, replay.plan_ranks(queries, docs, gold, examples))
    elif args.mode == "rank":
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        assert list(plan) == random.Random(replay.RANK_SEED).sample(sorted(examples), replay.RANK_COUNT)
        jobs = [(qid, entry["top"], entry["r"]) for qid, entry in plan.items()]
        replay.capture(args.output, jobs, queries, docs, gold, examples)
        replay.summary(args.output)
    else:
        performance(args.output, args.plan, args.repeats, queries, docs, gold, examples)


if __name__ == "__main__":
    main()
