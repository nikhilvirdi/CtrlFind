"""Benchmark the saved training panel with either stock or job-contained execution."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
from time import perf_counter, sleep
from unittest.mock import patch

from verification_optimization import baseline, metrics, stock, workload
from src import execute


def profile_supervisors(output):
    """Measure pooling supervisors while every candidate still gets a fresh job."""
    codes = ["pass"] * 50
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=12) as pool:
        ready = [pool.submit(sleep, .2) for _ in range(12)]
        for worker in ready:
            worker.result()
        data = {"workers": 12, "supervisor_start_seconds": perf_counter() - start,
                "threads_seconds": [], "supervisors_seconds": []}

        def threads():
            assert all(execute.verify(codes, ("", ""), workers=12))

        def supervisors():
            assert list(pool.map(execute.run_snippet, codes, ["\n"] * 50)) == [""] * 50

        for order in ((threads, supervisors), (supervisors, threads), (threads, supervisors)):
            for run in order:
                start = perf_counter()
                run()
                data[f"{run.__name__}_seconds"].append(perf_counter() - start)
    baseline.write_json(output, data)
    print(data, flush=True)


def run_panel(output, mode, repeats):
    saved, queries, docs, gold, examples = workload()
    expected = next(run["passer_ids"] for run in saved["runs"] if run["mode"] == "cold")
    assert all(run["passer_ids"] == expected for run in saved["runs"] if run["mode"] == "cold")

    runner = execute._run_candidate if mode == "job" else stock

    with tempfile.TemporaryDirectory(prefix="verifier_containment_") as scratch:
        root = Path(scratch)
        plan = root / "plan.json"
        baseline.write_json(plan, {qid: {"top": row["top"], "r": row["r"]} for qid, row in saved["plan"].items()})
        raw = root / "results.json"
        with patch.object(execute, "_run_candidate", runner):
            baseline.performance(raw, plan, repeats, queries, docs, gold, examples)
        evidence = json.loads(raw.read_text(encoding="utf-8"))
    evidence["runner"] = mode
    job_source = Path(execute.__file__).with_name("_windows_job.py")
    evidence["windows_job_sha256"] = hashlib.sha256(job_source.read_bytes()).hexdigest() if mode == "job" else None
    evidence["reference_metrics"] = metrics(saved["plan"], expected)
    for run in evidence["runs"]:
        if run["mode"] != "cold":
            continue
        actual = run["passer_ids"]
        run["metrics"] = metrics(saved["plan"], actual)
        assert actual.keys() == expected.keys()
        run["differences"] = {qid: sorted(set(actual[qid]) ^ set(expected[qid]))
                              for qid in actual if actual[qid] != expected[qid]}
        print(mode, run["repeat"], run["metrics"], run["differences"], flush=True)
    baseline.write_json(output, evidence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stock", "job", "supervisors"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.mode == "supervisors":
        profile_supervisors(args.output)
    else:
        run_panel(args.output, args.mode, args.repeats)
