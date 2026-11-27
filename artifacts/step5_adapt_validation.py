"""Validate live verification using the frozen Step 3 training-only rankings."""

import argparse
from collections import Counter
import ctypes
import gzip
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import local
import time
from types import SimpleNamespace
from unittest.mock import patch

import verification_depth as depth
import verification_depth_v3 as v3
from src import execute
from src.mteb_wrappers import LiveExecReranker, SearchEncoderWrapper

audit = depth.audit
ARMS = ("K50", "K200", "ADAPT")
SEED = 20261125


def evidence():
    frozen = v3.check_frozen()
    post_stop = audit.read_json(audit.HERE / "verification_depth_v3_post_stop_audit.json")
    for filename, expected in post_stop["archived_artifact_sha256"].items():
        assert audit.sha_file(v3.RUN / filename) == expected, filename
    assert v3.protocol()["sha256"] == post_stop["registration_sha256"]
    return {"frozen": frozen, "v3": post_stop["archived_artifact_sha256"]}


def inputs():
    audit.configure()
    archived = audit.read_json(v3.RUN / "plan.json")
    plan, _, docs, hashes = depth.prepare(v3.RUN, persist=False)
    for key in ("dataset_revision", "rows", "halves", "full_corpus", "baseline_sha256", "split_sha256"):
        assert json.loads(json.dumps(plan[key])) == archived[key], key
    queries = audit.inputs()[0]
    by_id = {}
    for did, code in docs.items():
        candidate = depth.Candidate(code)
        candidate.did = did
        by_id[did] = candidate
    return archived, plan["runtime"], queries, by_id, hashes


def runner(folder, arm, docs):
    live = LiveExecReranker(SimpleNamespace(mteb_model_meta=None),
                            folder / (("K50" if arm == "ADAPT" else arm) + ".jsonl"),
                            k=200 if arm == "K200" else 50, adapt=arm == "ADAPT")
    live.doc_text = docs
    return live


def search(live, qid, row, queries):
    hits = {did: -pos / 200 for pos, did in enumerate(row["top"])}
    with patch.object(SearchEncoderWrapper, "search", return_value={qid: hits}):
        result = live.search({"id": [qid], "text": [queries[qid]]}, top_k=200)[qid]
    return sorted(result, key=lambda did: (result[did], did), reverse=True)


def replay(plan, queries, docs, hashes):
    counts = {}
    with TemporaryDirectory(prefix="ctrlfind_replay_") as tmp:
        for rep in v3.REPS:
            with gzip.open(v3.RUN / f"rankings{rep}.json.gz", "rt", encoding="ascii") as stream:
                rankings = json.load(stream)
            counts[rep] = {arm: {"queries": 0, "attempts": 0, "passes": 0} for arm in ("K50", "ADAPT")}
            runners = {arm: runner(Path(tmp), arm, docs) for arm in counts[rep]}
            for qid, batch in v3.read_rep(v3.RUN, plan, hashes, rep):
                row = plan["rows"][qid]
                for arm, live in runners.items():
                    reference = rankings[qid][arm]
                    records = batch["records"][:reference["attempts"]]
                    calls = []

                    def stored(codes, example, timeout, *, workers, allow_trailing_padding, strict=False):
                        offset = len(calls)
                        ids = [code.did for code in codes]
                        calls.extend(ids)
                        expected = records[offset:len(calls)]
                        assert ids == [entry["id"] for entry in expected], (rep, qid, arm)
                        assert list(example) == row["example"] and allow_trailing_padding == row["padding"]
                        assert timeout == 1 and workers == depth.WORKERS and strict == (arm == "ADAPT")
                        return [entry["passed"] for entry in expected]

                    with patch("src.mteb_wrappers.verify", side_effect=stored):
                        order = search(live, qid, row, queries)
                    assert order == reference["top200"], (rep, qid, arm)
                    assert calls == [entry["id"] for entry in records], (rep, qid, arm)
                    passes = [entry["id"] for entry in records if entry["passed"]]
                    assert passes == reference["passes"], (rep, qid, arm)
                    count = counts[rep][arm]
                    count["queries"] += 1
                    count["attempts"] += len(calls)
                    count["passes"] += len(passes)
                    with patch("src.mteb_wrappers.verify", side_effect=AssertionError("Replay checkpoint miss")):
                        assert search(live, qid, row, queries) == order
            for live in runners.values():
                live.checkpoint.unlink()
    return {"replicates": counts, "candidate_launches": 0}


def replay_measured(folder, plan, queries, docs):
    counts = {}
    with TemporaryDirectory(prefix="ctrlfind_measured_replay_") as tmp:
        for rep in v3.REPS:
            with gzip.open(folder / f"rep{rep}_cold.json.gz", "rt", encoding="ascii") as stream:
                ledger = json.load(stream)
            counts[rep] = {arm: {"completed": 0, "aborted": 0, "verification_slots": 0} for arm in ARMS}
            runners = {arm: runner(Path(tmp), arm, docs) for arm in ARMS}
            for qid, arms in ledger.items():
                row = plan["rows"][qid]
                for arm, observed in arms.items():
                    live = runners[arm]
                    pending = iter(observed["batches"])
                    records = {r["id"]: r for r in observed["records"]}
                    stats = counts[rep][arm]

                    def stored(codes, example, timeout, *, workers, allow_trailing_padding, strict=False):
                        ids = [candidate.did for candidate in codes]
                        assert ids == next(pending) and list(example) == row["example"]
                        assert timeout == 1 and workers == depth.WORKERS and allow_trailing_padding == row["padding"]
                        assert strict == (arm == "ADAPT")
                        stats["verification_slots"] += len(ids)
                        if any("passed" not in records[did] for did in ids):
                            raise execute.VerificationInfrastructureError("archived execution error")
                        return [records[did]["passed"] for did in ids]

                    with patch("src.mteb_wrappers.verify", side_effect=stored):
                        try:
                            order = search(live, qid, row, queries)
                        except execute.VerificationInfrastructureError as exc:
                            assert observed["error"] == str(exc), (rep, qid, arm)
                            stats["aborted"] += 1
                        else:
                            assert observed["error"] is None and order == observed["order"], (rep, qid, arm)
                            stats["completed"] += 1
                    assert next(pending, None) is None
            for live in runners.values():
                live.checkpoint.unlink()
    return {"replicates": counts, "candidate_launches": 0}


def observe(live, qid, row, queries):
    """Measure production reranker calls while recording results at the launch boundary."""
    launch, snippet, verify = execute._run_candidate, execute.run_snippet, execute.verify
    context = local()
    records, batches = [], []

    def run(*args, **kwargs):
        record = context.record
        record["launched"] = True
        start = time.perf_counter()
        try:
            result = launch(*args, **kwargs)
            record["phase"] = "exit_zero" if result.returncode == 0 else "nonzero"
            record["exit_code"] = result.returncode
            if result.stdout is None or result.stderr is None:
                record["phase"] = "pipe_capture_error"
            return result
        except subprocess.TimeoutExpired:
            record["phase"] = "timeout"
            raise
        except Exception as exc:
            record.update(phase="infrastructure", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["seconds"] = time.perf_counter() - start

    def capture(code, *args, **kwargs):
        context.record = batch[code.did]
        context.record["phase"] = "setup_error"
        try:
            output = snippet(code, *args, **kwargs)
            if output is not None:
                context.record["stdout_sha256"] = depth.digest(output)
            return output
        except execute.VerificationInfrastructureError:
            context.record["strict_failure"] = True
            raise

    def inspect(codes, *args, **kwargs):
        nonlocal batch
        batch = {code.did: {"id": code.did, "phase": "cancelled", "launched": False} for code in codes}
        records.extend(batch.values())
        batches.append(list(batch))
        flags = verify(codes, *args, **kwargs)
        for item, passed in zip(batch.values(), flags, strict=True):
            item["passed"] = passed
        return flags

    batch = {}
    order = error = None
    with (patch.object(execute, "_run_candidate", run), patch.object(execute, "run_snippet", capture),
          patch("src.mteb_wrappers.verify", inspect)):
        start, cpu = time.perf_counter(), time.process_time()
        try:
            order = search(live, qid, row, queries)
        except execute.VerificationInfrastructureError as exc:
            error = str(exc)
        elapsed, cpu_seconds = time.perf_counter() - start, time.process_time() - cpu
    rank = None
    if order is not None:
        rank = order.index(row["gold"]) + 1 if row["gold"] in order else row["rank"]
    return {"seconds": elapsed, "parent_cpu_seconds": cpu_seconds, "parent_rss_bytes": parent_rss(),
            "batches": batches, "records": records, "order": order, "rank": rank, "error": error}


def parent_rss():
    values = (ctypes.c_size_t * 10)()
    values[0] = ctypes.sizeof(values)
    # GetCurrentProcess returns the pointer-sized pseudo-handle (HANDLE)-1.
    try:
        available = ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(values), values[0])
    except OSError:
        return None
    if not available:
        return None
    return int(values[2])  # PROCESS_MEMORY_COUNTERS.WorkingSetSize on x64


def timing(rows):
    values = sorted(row["seconds"] for row in rows)
    return {"mean": statistics.mean(values), "p50": statistics.median(values),
            "p95_nearest_rank": values[math.ceil(0.95 * len(values)) - 1], "sum": sum(values),
            "parent_cpu_seconds": sum(row["parent_cpu_seconds"] for row in rows),
            "parent_rss_max_bytes": max((row["parent_rss_bytes"] or 0 for row in rows), default=0)}


def summarize(rows, panel, plan):
    result = {}
    for arm in ARMS:
        cold = [rows[qid][arm]["cold"] for qid in panel]
        warm = [rows[qid][arm]["warm"] for qid in panel]
        records = [entry for row in cold for entry in row["records"]]
        non_gold = sum(entry.get("passed", False) and entry["id"] != plan["rows"][qid]["gold"]
                       for qid, row in zip(panel, cold, strict=True) for entry in row["records"])
        result[arm] = {"cold": timing(cold), "warm": timing(warm), "attempts": len(records),
                       "launches": sum(entry["launched"] for entry in records),
                       "phases": dict(Counter(entry["phase"] for entry in records)), "non_gold_passes": non_gold,
                       "non_gold_pass_rate": non_gold * 100 / len(records),
                       "cold_failures": [q for q, row in zip(panel, cold, strict=True) if row["error"]],
                       "warm_failures": [q for q, row in zip(panel, warm, strict=True) if row["error"]],
                       "warm_checkpoint_queries": sum(not row["batches"] for row in warm),
                       "warm_launches": sum(entry["launched"] for row in warm for entry in row["records"]),
                       "warm_order_differences": [q for q, c, w in zip(panel, cold, warm, strict=True)
                                                  if c["order"] != w["order"]],
                       "extensions": sum(len(row["batches"]) == 2 for row in cold)}
        for half in ("A", "B", "pooled"):
            ids = [q for q in panel if half == "pooled" or plan["rows"][q]["half"] == half]
            complete = [q for q in ids if all(rows[q][a]["cold"]["rank"] is not None for a in ARMS)]
            ranks = [rows[q][arm]["cold"]["rank"] for q in complete]
            control = [rows[q]["K50"]["cold"]["rank"] for q in complete]
            metrics = depth.np.array([depth.metric(rank) for rank in ranks])
            delta = metrics[:, 0] - [depth.metric(rank)[0] for rank in control]
            draws = depth.np.random.default_rng(SEED).integers(0, len(complete), (10000, len(complete)))
            result[arm][half] = {"n": len(complete), "planned": len(ids), "ndcg": float(metrics[:, 0].mean() * 100),
                                 "mrr": float(metrics[:, 1].mean() * 100), "hits10": int(metrics[:, 2].sum()),
                                 "delta_ndcg": float(delta.mean() * 100),
                                 "delta_ci": depth.np.percentile(delta[draws].mean(axis=1)*100, [2.5, 97.5]).tolist(),
                                 "gold_rank_wins": sum(a < b for a, b in zip(ranks, control)),
                                 "gold_rank_losses": sum(a > b for a, b in zip(ranks, control)),
                                 "top10_exits": [q for q, a, b in zip(complete, ranks, control) if a > 10 >= b]}
    a, k = result["ADAPT"], result["K200"]
    result["gates"] = {
        "full_pooled_gain_and_ci": a["pooled"]["n"] == len(panel) and
        a["pooled"]["delta_ndcg"] >= 1 and a["pooled"]["delta_ci"][0] > 0,
        "both_halves_positive": min(a[h]["delta_ndcg"] for h in ("A", "B")) > 0,
        "non_gold_rate": a["non_gold_pass_rate"] <= 0.5,
        "no_top10_exits": not a["pooled"]["top10_exits"],
        "mean_and_p95_at_most_k200": all(a["cold"][m] <= k["cold"][m] for m in ("mean", "p95_nearest_rank")),
        "no_adapt_infrastructure_failure": not (a["cold_failures"] or a["warm_failures"]),
    }
    return result


def benchmark(destination, plan, runtime, queries, docs):
    halves = {half: random.Random(f"{SEED}:step5:{half}").sample(
        sorted(q for q in plan["halves"][half] if plan["rows"][q]["example"]), 60)
        for half in ("A", "B")}
    panel = halves["A"] + halves["B"]
    audit.write_json(destination / "plan.json", {"halves": halves, "order": panel, "runtime": runtime,
                                               "cpu": platform.processor(), "torch_threads": audit.THREADS,
                                               "torch_interop_threads": 1, "arms": ARMS,
                                               "rows": {q: plan["rows"][q] for q in panel}})
    all_rows, summaries = {}, {}
    for rep in (1, 2):
        rows = {q: {arm: {} for arm in ARMS} for q in panel}
        with TemporaryDirectory(prefix=f"ctrlfind_adapt_{rep}_") as tmp:
            runners = {arm: runner(Path(tmp), arm, docs) for arm in ARMS}
            for phase in ("cold", "warm"):
                for i, qid in enumerate(panel):
                    offset = (i + rep - 1) % len(ARMS)
                    for arm in ARMS[offset:] + ARMS[:offset]:
                        rows[qid][arm][phase] = observe(runners[arm], qid, plan["rows"][qid], queries)
                    if (i + 1) % 20 == 0:
                        print(f"rep {rep} {phase}: {i+1}/{len(panel)}", flush=True)
                v3.compressed_json(destination / f"rep{rep}_{phase}.json.gz",
                                   {q: {a: data[phase] for a, data in arms.items()} for q, arms in rows.items()})
        all_rows[rep] = rows
        summaries[rep] = summarize(rows, panel, plan)
        audit.write_json(destination / f"rep{rep}_summary.json", summaries[rep])
    noise = {}
    for qid in panel:
        for arm in ARMS:
            records = [{entry["id"]: entry for entry in all_rows[rep][qid][arm]["cold"]["records"]} for rep in (1, 2)]
            shared = [key for key in records[0] if key in records[1]]
            v3.compare_records(qid, [records[0][key] for key in shared if all("passed" in r[key] for r in records)],
                               [records[1][key] for key in shared if all("passed" in r[key] for r in records)], noise, arm)
    return {"by_repetition": summaries, "paired_repeat_noise": noise}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--benchmark", action="store_true")
    mode.add_argument("--replay-measured", type=Path, metavar="BENCHMARK_OUTPUT")
    args = parser.parse_args()
    before = evidence()
    plan, runtime, queries, docs, hashes = inputs()
    args.destination.mkdir()
    receipt = replay(plan, queries, docs, hashes)
    audit.write_json(args.destination / "archive_replay.json", receipt)
    print(f"Archive replay matched: {receipt}", flush=True)
    if args.benchmark:
        audit.write_json(args.destination / "results.json", benchmark(args.destination, plan, runtime, queries, docs))
    if args.replay_measured:
        observed = replay_measured(args.replay_measured, plan, queries, docs)
        audit.write_json(args.destination / "measured_replay.json", observed)
        print(f"Measured replay matched: {observed}", flush=True)
    assert evidence() == before
    audit.write_json(args.destination / "integrity.json", before)


if __name__ == "__main__":
    main()
