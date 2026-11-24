"""Profile and replay CtrlFind's fixed 200-query execution performance workload."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter, sleep
from unittest.mock import patch
import warnings

import verification_performance as baseline
from verification_research import SNAPSHOT, decode, load_train
from src import execute

BASELINE = Path(__file__).with_name("verification_performance_train.json")


def stock(args, **kwargs):
    return subprocess.run(args, capture_output=True, text=True, **kwargs)


def syntax_invalid(code):
    try:
        compile(code.encode("utf-8"), "solution.py", "exec", dont_inherit=True)
    except SyntaxError:
        return True
    return False


def workload():
    saved = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert saved["revision"] == SNAPSHOT.name and saved["k"] == 50
    queries, docs, gold = load_train(SNAPSHOT)
    examples = {qid: ex for qid, query in queries.items() if (ex := execute.parse_example(query))}
    assert len(examples) == 511 and set(saved["plan"]) <= examples.keys()
    return saved, queries, docs, gold, examples


def profile(data, output):
    saved, _, docs, _, examples = data
    sources = [docs[did] for row in saved["plan"].values() for did in row["top"]]
    start = perf_counter()
    invalid = [syntax_invalid(source) for source in sources]
    all_compile_seconds = perf_counter() - start
    distinct = list(dict.fromkeys(sources))
    start = perf_counter()
    distinct_invalid = sum(syntax_invalid(source) for source in distinct)
    distinct_compile_seconds = perf_counter() - start

    phases = defaultdict(list)
    original_create, original_communicate = subprocess.Popen.__init__, subprocess.Popen.communicate
    original_dir, original_cleanup = tempfile.TemporaryDirectory.__init__, tempfile.TemporaryDirectory.cleanup
    original_open = open
    original_runner = execute.run_snippet

    def timed(label, fn):
        def call(*args, **kwargs):
            start = perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                phases[label].append(perf_counter() - start)
        return call

    # No-op subprocesses bound interpreter startup with current stdio and filesystem setup.
    with (
        patch.object(subprocess.Popen, "__init__", timed("create_process", original_create)),
        patch.object(subprocess.Popen, "communicate", timed("communicate", original_communicate)),
        patch.object(tempfile.TemporaryDirectory, "__init__", timed("mkdir", original_dir)),
        patch.object(tempfile.TemporaryDirectory, "cleanup", timed("cleanup", original_cleanup)),
        patch.object(execute, "open", timed("open_source", original_open), create=True),
        patch.object(execute, "run_snippet", timed("whole_candidate", original_runner)),
        patch.object(execute, "_run_candidate", stock),
    ):
        start = perf_counter()
        assert all(execute.verify(["pass"] * 50, ("", ""), workers=12))
        noop_wall = perf_counter() - start
        noop = dict(phases)
        phases.clear()
        start = perf_counter()
        for qid, row in list(saved["plan"].items())[:12]:
            execute.verify([docs[did] for did in row["top"]], examples[qid], workers=12)
        real_wall = perf_counter() - start
        real = dict(phases)

    def describe(sample):
        return {key: {"count": len(values), "median_ms": statistics.median(values) * 1000,
                      "p95_ms": sorted(values)[math.ceil(.95 * len(values)) - 1] * 1000}
                for key, values in sample.items()}

    result = dict(pair_count=len(sources), unique_sources=len(distinct), invalid_pairs=sum(invalid),
                  invalid_sources=distinct_invalid, all_compile_seconds=all_compile_seconds,
                  unique_compile_seconds=distinct_compile_seconds,
                  noop_wall=noop_wall, noop=describe(noop),
                  real_wall=real_wall, real=describe(real))
    baseline.write_json(output, result)
    print(json.dumps(result), flush=True)


def startup(output):
    """Isolate interpreter setup with bounded synthetic candidate programs."""
    if sys.platform != "win32" or sys.prefix == sys.base_prefix:
        raise SystemExit("launcher controls require a Windows virtual environment")
    real_run = subprocess.run

    def no_site(args, **kwargs):
        return real_run([args[0], "-S", *args[1:]], **kwargs)

    def direct(args, **kwargs):
        env = {**os.environ, "__PYVENV_LAUNCHER__": sys.executable}
        return real_run(args, executable=sys._base_executable, env=env, **kwargs)

    launchers = {"stock": real_run, "no_site": no_site, "direct": direct}
    result = {name: {"noop_k50_seconds": []} for name in launchers}
    for order in (list(launchers), list(reversed(launchers)), list(launchers)[1:] + ["stock"]):
        for name in order:
            def launch(args, **kwargs):
                return launchers[name](args, capture_output=True, text=True, **kwargs)

            with patch.object(execute, "_run_candidate", launch):
                start = perf_counter()
                assert all(execute.verify(["pass"] * 50, ("", ""), workers=12))
                result[name]["noop_k50_seconds"].append(perf_counter() - start)

    inspect = ("import json, os, sys; print(json.dumps({'site': 'site' in sys.modules, "
               "'venv': sys.base_prefix != sys.prefix, 'direct_parent': os.getppid() == "
               f"{os.getpid()}}}))")
    for name, run in launchers.items():
        result[name]["environment"] = json.loads(run([sys.executable, "-c", inspect], capture_output=True,
                                                      text=True, check=True).stdout)
        with tempfile.TemporaryDirectory(prefix="verifier_timeout_probe_") as directory:
            child = "from pathlib import Path; import time; time.sleep(0.6); Path('marker').touch()"
            parent = ("import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', "
                      f"{child!r}], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
                      "stderr=subprocess.DEVNULL); print('spawned', flush=True); time.sleep(1.5)")
            start = perf_counter()
            try:
                run([sys.executable, "-c", parent], cwd=directory, timeout=0.4, text=True, capture_output=True)
                raise AssertionError("bounded timeout probe completed unexpectedly")
            except subprocess.TimeoutExpired as exc:
                observation = {"elapsed": perf_counter() - start, "spawned": "spawned" in (decode(exc.stdout) or "")}
            sleep(1.6)
            observation["descendant_wrote_after_timeout"] = (Path(directory) / "marker").exists()
            result[name]["timeout_probe"] = observation

    with tempfile.TemporaryDirectory(prefix="verifier_startup_probe_") as directory:
        root = Path(directory)
        (root / "sitecustomize.py").write_text("import os\nprint('ready', flush=True)\nos._exit(0)\n", encoding="utf-8")
        invalid = "return 1\n"
        path = root / "solution.py"
        path.write_text(invalid, encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(root)}
        observed = real_run([sys.executable, path], input="\n", cwd=root, capture_output=True,
                            timeout=1, text=True, env=env)
        result["syntax_startup_control"] = {"compile_invalid": syntax_invalid(invalid),
                                           "actual_pass": observed.returncode == 0 and observed.stdout.strip() == "ready"}
    baseline.write_json(output, result)
    print(json.dumps(result), flush=True)


def metrics(plan, verdicts):
    ranks = []
    for qid, row in plan.items():
        passed = set(verdicts[qid])
        ranking = sorted(row["top"], key=lambda did: did not in passed)
        ranks.append(ranking.index(row["gold"]) + 1 if row["gold"] in ranking else row["r"])
    return {"ndcg_at_10": 100 * sum(1 / math.log2(rank + 1) for rank in ranks if rank <= 10) / len(ranks),
            "mrr_at_10": 100 * sum(1 / rank for rank in ranks if rank <= 10) / len(ranks)}


def sweep(data, output, conditions):
    saved, queries, docs, gold, examples = data
    plan = saved["plan"]
    expected = next(run["passer_ids"] for run in saved["runs"] if run["mode"] == "cold")
    assert all(run["passer_ids"] == expected for run in saved["runs"] if run["mode"] == "cold")
    original_runner = execute.run_snippet
    with tempfile.TemporaryDirectory(prefix="verifier_sweep_") as temp:
        root = Path(temp)
        plan_file = root / "plan.json"
        baseline.write_json(plan_file, {qid: {"top": row["top"], "r": row["r"]} for qid, row in plan.items()})
        with output.open("x", encoding="utf-8") as summary:
            for trial, condition in enumerate(conditions):
                parts = condition.split(":")
                workers = int(parts[0])
                mode = parts[1] if len(parts) == 2 else "stock"
                skipped = []
                used_compile = {}

                def run(code, stdin, timeout):
                    if mode == "syntax":
                        flag = used_compile.get(code)
                        if flag is None:
                            flag = syntax_invalid(code)
                            used_compile[code] = flag
                        if flag:
                            skipped.append(None)
                            return None
                    return original_runner(code, stdin, timeout)

                run_file = root / f"trial{trial}.json"
                with (
                    patch.object(os, "cpu_count", return_value=workers),
                    patch.object(execute, "run_snippet", run),
                    patch.object(execute, "_run_candidate", stock),
                ):
                    baseline.performance(run_file, plan_file, 1, queries, docs, gold, examples)
                run_data = json.loads(run_file.read_text(encoding="utf-8"))
                cold, warm = run_data["runs"]
                got = cold["passer_ids"]
                differences = {
                    qid: {"lost": sorted(set(expected[qid]) - set(got[qid])),
                          "gained": sorted(set(got[qid]) - set(expected[qid]))}
                    for qid in got if expected[qid] != got[qid]
                }
                assert sum(cold["process_counts"].values()) + len(skipped) == 10000
                row = dict(workers=workers, mode=mode, syntax_skips=len(skipped), cold=cold, warm=warm,
                           differences=differences, metrics=metrics(plan, got),
                           saved_metrics=metrics(plan, expected), machine=run_data["machine"])
                summary.write(json.dumps(row, separators=(",", ":")) + "\n")
                summary.flush()
                print(condition, row["metrics"], differences, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("conditions", nargs="*", help="workers[:stock|syntax], or omit to profile")
    parser.add_argument("--startup", action="store_true", help="run Windows interpreter startup controls")
    args = parser.parse_args()
    if args.startup:
        startup(args.output)
    else:
        data = workload()
        with warnings.catch_warnings(action="ignore", category=SyntaxWarning):
            if args.conditions:
                sweep(data, args.output, args.conditions)
            else:
                profile(data, args.output)
