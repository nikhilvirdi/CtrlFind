"""Preregistered Codeforces verification-depth experiment. No production writes."""
from __future__ import annotations

import argparse
from collections import Counter
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
import threading
import time
from unittest.mock import patch

import numpy as np

import retrieval_consistency as audit
from src import execute
from src.config import EXEC_TIMEOUT


ARMS = ("K50", "K100", "K200", "ADAPT")
TITLE = "### Execution-verification depth preregistration"
BASELINE = audit.HERE / "dev_stdin_full_baseline.json"
SPLIT = audit.HERE / "dev_stdin_ab_split.json"
SOURCE = "codeforces.com"
WORKERS = os.cpu_count() or 2


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def registration():
    log = (audit.ROOT / "docs" / "research-log.md").read_text(encoding="utf-8")
    section = log.split(TITLE + "\n", 1)[1].split("\n### ", 1)[0].strip()
    assert "2026-11-25 03:22:32 +05:30" in section
    return {"text": TITLE + "\n\n" + section, "sha256": digest(section)}


def runtime():
    return {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "timeout": EXEC_TIMEOUT, "workers": WORKERS,
        "verifier_sha256": audit.sha_file(Path(execute.__file__)),
        "windows_job_sha256": audit.sha_file(Path(execute._windows_job.__file__)),
        "encoding": locale.getpreferredencoding(False),
        "utf8_mode": sys.flags.utf8_mode,
        "inherited_environment_sha256": audit.json_digest(sorted(os.environ.items())),
    }


def prepare(folder, *, persist=True):
    assert os.name == "nt"
    queries, sources, gold, _, dev, old, ablation = audit.inputs()
    baseline, split = audit.read_json(BASELINE), audit.read_json(SPLIT)
    assert baseline["config"]["query_ids"] == dev
    assert split["source_panel_sha256"] == audit.ids_digest(dev) and split["seed"] == audit.SEED
    assert audit.sha_file(SPLIT) == baseline["config"]["split_file_sha256"]
    assert audit.sha_file(audit.parquet("data/train-*.parquet")) == baseline["config"]["training_qrels_sha256"]
    halves = {half: [qid for qid in split["halves"][half]["query_ids"] if sources[qid] == SOURCE]
              for half in ("A", "B")}
    assert list(map(len, halves.values())) == [238, 239]
    assert not set(halves["A"]) & set(halves["B"])
    assert set(halves["A"] + halves["B"]) == {qid for qid in dev if sources[qid] == SOURCE}
    timing = {half: random.Random(f"{audit.SEED}:depth-timing:{half}").sample(sorted(ids), 20)
              for half, ids in halves.items()}
    corpus_file = audit.parquet("corpus/*.parquet")
    config = baseline["config"]["candidate_corpus"]
    assert audit.sha_file(corpus_file) == config["parquet_sha256"] == old["corpus"]["parquet_sha256"]
    snippets = audit.pq.read_table(corpus_file, columns=["_id", "text"]).to_pylist()
    docs = {row["_id"]: row["text"] for row in snippets}
    assert len(docs) == len(snippets) == config["total"] == 8765
    assert list(docs) == config["document_ids_in_parquet_order"]
    hashes = {did: digest(code) for did, code in docs.items()}
    rows = {}
    for half, ids in halves.items():
        for qid in ids:
            archived = baseline["rows"][qid]
            top, did = archived["top200"], gold[qid]
            assert archived["source"] == SOURCE and archived["gold_id"] == did
            assert len(top) == len(set(top)) == 200 and set(top) <= docs.keys()
            assert (top.index(did) + 1 if did in top else None) == (archived["rank"] if archived["rank"] <= 200 else None)
            rows[qid] = {"half": half, "gold": did, "rank": archived["rank"], "top": top,
                         "example": execute.parse_example(queries[qid]),
                         "padding": execute.allows_trailing_padding(queries[qid]),
                         "query_sha256": digest(queries[qid])}
    assert sum(row["example"] is not None for row in rows.values()) == 474
    plan = {"protocol": registration(), "dataset_revision": audit.REVISION, "seed": audit.SEED,
            "baseline_sha256": audit.sha_file(BASELINE), "split_sha256": audit.sha_file(SPLIT),
            "full_corpus": config, "runtime": runtime(), "halves": halves, "timing": timing, "rows": rows}
    if not persist:
        return plan, baseline, docs, hashes
    path = folder / "plan.json"
    if path.exists():
        assert audit.read_json(path) == json.loads(json.dumps(plan)), "Study inputs or runtime changed"
    else:
        audit.write_json(path, plan)
    return plan, baseline, docs, hashes


class Candidate(str):
    """A distinct object per slot, including slots with duplicate code."""


def observe(ids, docs, row, *, time_completions=False):
    codes = [Candidate(docs[did]) for did in ids]
    evidence = {id(code): {"id": did} for did, code in zip(ids, codes, strict=True)}
    local = threading.local()
    snippet, launch = execute.run_snippet, execute._run_candidate

    def run_snippet(code, *args):
        local.record = evidence[id(code)]
        return snippet(code, *args)

    def run(*args, **kwargs):
        record = local.record
        start = time.perf_counter()
        record["phase"] = "launch_error"
        try:
            result = launch(*args, **kwargs)
            record.update(phase="exit_zero" if result.returncode == 0 else "nonzero",
                          exit_code=result.returncode, stdout_length=len(result.stdout or ""),
                          stderr=(result.stderr or "")[-500:])
            local.stdout = result.stdout or ""
            return result
        except subprocess.TimeoutExpired:
            record["phase"] = "timeout"
            raise
        except Exception as error:
            record["error"] = repr(error)
            raise
        finally:
            record["launch_seconds"] = time.perf_counter() - start

    def capture(code, *args):
        local.stdout = ""
        result = run_snippet(code, *args)
        record = local.record
        if result is not None:
            record["stdout_sha256"] = digest(local.stdout)
            # A matching output is retained below from the fixture and the verifier.
        record.setdefault("phase", "setup_error")
        if time_completions:
            record["completed_after_seconds"] = time.perf_counter() - start
        return result

    start = time.perf_counter()
    with patch.object(execute, "_run_candidate", run), patch.object(execute, "run_snippet", capture):
        verdicts = execute.verify(codes, tuple(row["example"]), timeout=EXEC_TIMEOUT, workers=WORKERS,
                                  allow_trailing_padding=row["padding"])
    elapsed = time.perf_counter() - start
    records = list(evidence.values())
    for record, verdict in zip(records, verdicts, strict=True):
        record["passed"] = verdict
        assert not verdict or record["phase"] == "exit_zero"
    return {"seconds": elapsed, "records": records}


def cache_key(plan, qid, ids, hashes, scope):
    row = plan["rows"][qid]
    return audit.json_digest([qid, scope, row["query_sha256"], row["example"], row["padding"],
                              plan["runtime"], [[did, hashes[did]] for did in ids]])


def run_cached(path, plan, qid, ids, docs, hashes, scope):
    key = cache_key(plan, qid, ids, hashes, scope)
    if path.exists():
        start = time.perf_counter()
        saved = audit.read_json(path)
        assert saved["fingerprint"] == key and [r["id"] for r in saved["records"]] == ids
        assert all(not r["passed"] or r["phase"] == "exit_zero" for r in saved["records"])
        return saved, "hit", time.perf_counter() - start
    if ids:
        assert plan["rows"][qid]["example"]
        saved = observe(ids, docs, plan["rows"][qid])
    else:
        saved = {"seconds": 0.0, "records": []}
    saved["fingerprint"] = key
    partial = path.with_suffix(".tmp")
    with partial.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(saved, output, ensure_ascii=True, separators=(",", ":"))
        output.write("\n")
    partial.replace(path)
    return saved, "miss", saved["seconds"]


def passes(records):
    return [record["id"] for record in records if record["passed"]]


def collect(folder, plan, docs, hashes):
    cache = folder / "shared"
    cache.mkdir(exist_ok=True)
    for half, qids in plan["halves"].items():
        for n, qid in enumerate(qids, 1):
            row = plan["rows"][qid]
            ids = row["top"] if row["example"] else []
            saved, state, _ = run_cached(cache / f"{qid}.json", plan, qid, ids, docs, hashes, "shared_K200")
            if n % 10 == 0 or n == len(qids):
                print(half, n, len(qids), state, qid, round(saved["seconds"], 2), len(passes(saved["records"])), flush=True)


def execute_timing(folder, plan, docs, hashes):
    cache = folder / "timed"
    cache.mkdir(exist_ok=True)
    timings = {}
    for half, qids in plan["timing"].items():
        for i, qid in enumerate(qids):
            row = plan["rows"][qid]
            order = ARMS[i % len(ARMS):] + ARMS[:i % len(ARMS)]
            shared = audit.read_json(folder / "shared" / f"{qid}.json")["records"]
            timings[qid] = {"half": half, "order": order, "arms": {}}
            for arm in order:
                n = 50 if arm == "ADAPT" else int(arm[1:])
                ids = row["top"][:n] if row["example"] else []
                record, state, _ = run_cached(cache / f"{qid}_{arm}_first.json", plan, qid, ids, docs,
                                              hashes, f"timing:{arm}:first")
                seconds = [record["seconds"]]
                fresh = record["records"]
                batch_states = [state]
                if arm == "ADAPT" and ids and not passes(fresh):
                    more, state, _ = run_cached(cache / f"{qid}_{arm}_extension.json", plan, qid, row["top"][50:],
                                                docs, hashes, f"timing:{arm}:extension")
                    fresh = fresh + more["records"]
                    seconds.append(more["seconds"])
                    batch_states.append(state)
                differences = [{"id": entry["id"], "fresh": entry["passed"], "shared": control["passed"],
                                "fresh_phase": entry["phase"], "shared_phase": control["phase"]}
                               for entry, control in zip(fresh, shared) if entry["passed"] != control["passed"]]
                timings[qid]["arms"][arm] = {"seconds": sum(seconds), "batch_seconds": seconds,
                                             "attempts": len(fresh), "passes": passes(fresh),
                                             "phase_counts": dict(Counter(r["phase"] for r in fresh)),
                                             "verdict_differences": differences, "resumed_batches": batch_states.count("hit")}
            print("timing", half, i + 1, qid,
                  {k: round(v["seconds"], 2) for k, v in timings[qid]["arms"].items()}, flush=True)
    audit.write_json(folder / "timing.json", timings)


def metric(rank):
    return (1 / math.log2(rank + 1), 1 / rank, 1) if rank <= 10 else (0.0, 0.0, 0)


def rank_row(row, records):
    top, gold = row["top"], row["gold"]
    result = {}
    for arm in ARMS:
        n = 50 if arm == "ADAPT" and passes(records[:50]) else 200 if arm == "ADAPT" else int(arm[1:])
        seen = records[:n]
        passing = passes(seen)
        pass_set = set(passing)
        order = [*passing, *(did for did in top if did not in pass_set)]
        rank = order.index(gold) + 1 if gold in top else row["rank"]
        result[arm] = {"rank": rank, "gold_position_among_passers": passing.index(gold) + 1 if gold in pass_set else None,
                       "attempts": len(seen), "passes": passing, "top200": order}
    return result


def replay(folder, plan, baseline, hashes):
    start = time.perf_counter()
    rows, per_query = {}, {}
    for qid, row in plan["rows"].items():
        ids = row["top"] if row["example"] else []
        t0 = time.perf_counter()
        raw = audit.read_json(folder / "shared" / f"{qid}.json")
        assert raw["fingerprint"] == cache_key(plan, qid, ids, hashes, "shared_K200")
        records = raw["records"]
        assert [record["id"] for record in records] == ids
        assert all(not r["passed"] or r["phase"] == "exit_zero" for r in records)
        rows[qid] = {"half": row["half"], "gold": row["gold"], "initial_rank": row["rank"],
                     "K200_observed_seconds": raw["seconds"], "arms": rank_row(row, records),
                     "phase_counts": dict(Counter(r["phase"] for r in records))}
        per_query[qid] = time.perf_counter() - t0
    return rows, {"total_seconds": time.perf_counter() - start, "per_query_seconds": per_query,
                  "shared_cache_queries": len(rows), "unique_candidate_verdicts": sum(sum(row["phase_counts"].values())
                                                                                      for row in rows.values())}


def latency_metrics(values):
    return {"mean": statistics.mean(values), "p95": sorted(values)[math.ceil(0.95 * len(values)) - 1],
            "total": sum(values), "queries": len(values)}


def summarize(rows, groups, timings, evidence):
    output = {}
    for group, qids in groups.items():
        analysis = {}
        control_ranks = [rows[qid]["arms"]["K50"]["rank"] for qid in qids]
        control_ndcg = np.array([metric(rank)[0] for rank in control_ranks])
        draws = np.random.default_rng(audit.SEED).integers(0, len(qids), size=(10_000, len(qids)))
        for arm in ARMS:
            results = [rows[qid]["arms"][arm] for qid in qids]
            ranks = [item["rank"] for item in results]
            ndcg, mrr, hit10 = map(np.asarray, zip(*(metric(rank) for rank in ranks)))
            deltas = ndcg - control_ndcg
            flags = Counter(evidence.get(qid, {}).get(did, {}).get("status", "unknown")
                            for qid, item in zip(qids, results, strict=True)
                            for did in item["passes"] if did != rows[qid]["gold"])
            attempts = sum(item["attempts"] for item in results)
            times = [v["arms"][arm]["seconds"] for qid, v in timings.items() if qid in qids]
            time_control = [v["arms"]["K50"]["seconds"] for qid, v in timings.items() if qid in qids]
            warm_cost = sum(len(item["passes"]) for item in results)
            verified_gold = [(item["rank"], item["gold_position_among_passers"]) for item in results
                             if item["gold_position_among_passers"] is not None]
            analysis[arm] = {
                "ndcg_at_10": float(ndcg.mean() * 100), "mrr_at_10": float(mrr.mean() * 100),
                "recall_at_10": audit.rate(int(hit10.sum()), len(qids)),
                "ndcg_delta": float(deltas.mean() * 100), "ndcg_delta_ci_95": (np.quantile(deltas[draws].mean(axis=1),
                                                                                           [0.025, 0.975]) * 100).tolist(),
                "mean_passers": warm_cost / len(qids), "passing_gold": len(verified_gold),
                "gold_rank_when_passing_mean": statistics.mean(x for x, _ in verified_gold) if verified_gold else None,
                "gold_rank_median": statistics.median(ranks), "gold_rank_mean": statistics.mean(ranks),
                "attempts": attempts, "incremental_attempts_vs_K50": attempts - sum(
                    rows[qid]["arms"]["K50"]["attempts"] for qid in qids),
                "warm_cached_verdicts": attempts, "warm_launches": 0,
                "non_gold_passes": dict(flags), "confirmed_false_rate": 100 * flags["false"] / attempts,
                "upper_false_rate": 100 * (flags["false"] + flags["unknown"]) / attempts,
                "rank_improved": [q for q, a, b in zip(qids, control_ranks, ranks, strict=True) if b < a],
                "rank_regressed": [q for q, a, b in zip(qids, control_ranks, ranks, strict=True) if b > a],
                "rank_unchanged": sum(a == b for a, b in zip(control_ranks, ranks, strict=True)),
                "cold_sample": latency_metrics(times) if times else None,
                "cold_sample_incremental_mean": statistics.mean(a - b for a, b in zip(times, time_control, strict=True))
                                                if times else None,
                "cold_sample_attempts": sum(v["arms"][arm]["attempts"] for qid, v in timings.items() if qid in qids),
                "fresh_verdict_differences": sum(len(v["arms"][arm]["verdict_differences"])
                                                  for qid, v in timings.items() if qid in qids),
            }
        analysis["K200"]["cold_census"] = latency_metrics([rows[qid]["K200_observed_seconds"] for qid in qids])
        output[group] = analysis
    return output


def finish(folder, plan, baseline, hashes, evidence_path):
    rows, warm = replay(folder, plan, baseline, hashes)
    repeat, warm_repeat = replay(folder, plan, baseline, hashes)
    assert rows == repeat
    groups = {**plan["halves"], "pooled": [*plan["halves"]["A"], *plan["halves"]["B"]]}
    timings = audit.read_json(folder / "timing.json") if (folder / "timing.json").exists() else {}
    evidence = audit.read_json(evidence_path)["pairs"] if evidence_path else {}
    summary = summarize(rows, groups, timings, evidence)
    stage1 = {}
    for source, stock in baseline["source_metrics"].items():
        stat = [metric(r["rank"]) for r in baseline["rows"].values() if r["source"] == source]
        stage1[source] = {**stock, "mrr_at_10": 100 * statistics.mean(x[1] for x in stat)}
    verdicts = {}
    for arm in ARMS[1:]:
        a, b, pooled = (summary[group][arm] for group in ("A", "B", "pooled"))
        verdicts[arm] = {
            "pooled_gain_and_ci": pooled["ndcg_delta"] >= 1 and pooled["ndcg_delta_ci_95"][0] > 0,
            "positive_in_both_halves": min(a["ndcg_delta"], b["ndcg_delta"]) > 0,
            "safety": "pass" if pooled["upper_false_rate"] <= 0.2 else
                      "fail" if pooled["confirmed_false_rate"] > 0.2 else "inconclusive",
        }
        v = verdicts[arm]
        v["qualifies"] = v["pooled_gain_and_ci"] and v["positive_in_both_halves"] and v["safety"] == "pass"
    for qid, row in rows.items():
        for arm in ARMS:
            row["arms"][arm]["rank_delta_vs_K50"] = row["arms"][arm]["rank"] - row["arms"]["K50"]["rank"]
    output = {"summaries": summary, "acceptance": verdicts, "stage1_only": stage1, "rows": rows,
              "deterministic": True, "rankings_sha256": audit.json_digest(rows), "warm_replay": [warm, warm_repeat],
              "evidence_sha256": audit.sha_file(evidence_path) if evidence_path else None}
    return output


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("collect", "timing", "summarize"))
    cli.add_argument("folder", type=Path)
    cli.add_argument("--evidence", type=Path)
    cli.add_argument("--output", default="results.json")
    args = cli.parse_args()
    args.folder.mkdir(exist_ok=True)
    plan, baseline, docs, hashes = prepare(args.folder)
    if args.mode == "collect":
        collect(args.folder, plan, docs, hashes)
    elif args.mode == "timing":
        execute_timing(args.folder, plan, docs, hashes)
    else:
        audit.write_json(args.folder / args.output, finish(args.folder, plan, baseline, hashes, args.evidence))
