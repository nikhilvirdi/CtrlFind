"""Registration v3. Only collect() executes candidates. No test queries or test labels."""
import argparse
import gzip
import json
import os
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import verification_depth as depth

audit = depth.audit
V2 = audit.HERE / "verification_depth_v2_run"
RUN = audit.HERE / "verification_depth_v3_run"
REPS = (1, 2)


def protocol():
    log = (audit.ROOT / "docs/research-log.md").read_text(encoding="utf-8")
    heading = "### Registration v3\n"
    assert log.count(heading) == 1
    body = log.split(heading)[1].split("\n### ")[0].split("\n## ")[0].strip()
    assert "79/23,700" in body and "twice, cold" in body
    return {"text": heading + "\n" + body, "sha256": depth.digest(body)}


def check_frozen():
    expected = audit.read_json(V2 / "frozen_before_sweep.json")
    for name, entry in expected["sha256"].items():
        assert audit.sha_file(audit.HERE / name) == entry["expected"], f"Frozen hash mismatch: {name}"
    row = audit.read_json(audit.HERE / "baseline_frozen.json")["scores"]["test"][0]
    assert [row["ndcg_at_10"], row["mrr_at_10"]] == expected["aggregate_scores"]
    return expected


def inputs(folder):
    audit.configure()
    assert audit.torch.get_num_threads() == 4 and audit.torch.get_num_interop_threads() == 1
    with patch.object(depth, "registration", protocol):
        plan, baseline, docs, hashes = depth.prepare(folder, persist=False)
    plan.pop("timing")  # v2's arm timing sample is not used in v3.
    controls = audit.read_json(V2 / "plan.json")
    for key in ("dataset_revision", "seed", "rows", "halves", "full_corpus", "baseline_sha256", "split_sha256"):
        assert json.loads(json.dumps(plan[key])) == controls[key], key
    for key in ("python", "platform", "timeout", "workers", "utf8_mode", "verifier_sha256", "windows_job_sha256"):
        assert plan["runtime"][key] == controls["runtime"][key], key
    return plan, baseline, docs, hashes


def identity(plan, replicate, qid, ids, hashes):
    return depth.cache_key(plan, qid, ids, hashes, f'{plan["protocol"]["sha256"]}:rep{replicate}:K200')


def collect(folder):
    assert not folder.exists(), "Cold collection requires a new output directory"
    frozen = check_frozen()
    plan, _, docs, hashes = inputs(folder)
    folder.mkdir()
    audit.write_json(folder / "plan.json", plan)
    audit.write_json(folder / "frozen_before.json", frozen)
    audit.write_json(folder / "machine.json", {
        "cpu": audit.read_json(V2 / "machine.json")["cpu"], "logical_cores": os.cpu_count(),
        "torch_threads": audit.torch.get_num_threads(), "torch_interop_threads": audit.torch.get_num_interop_threads(),
        "versions": audit.package_versions(), "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=audit.ROOT, text=True).strip(),
        "cache": "Empty, separate raw verdict stream per replicate. No read or reuse during collection.",
    })
    for replicate in REPS:
        started = time.perf_counter()
        attempts = timeouts = 0
        with (folder / f"cold{replicate}.jsonl.gz").open("xb") as stream:
            for i, (qid, row) in enumerate(plan["rows"].items(), 1):
                ids = row["top"] if row["example"] else []
                batch = depth.observe(ids, docs, row, time_completions=True) if ids else {"seconds": 0.0, "records": []}
                batch.update(qid=qid, fingerprint=identity(plan, replicate, qid, ids, hashes))
                assert [c["id"] for c in batch["records"]] == ids
                data = (json.dumps(batch, ensure_ascii=True, separators=(",", ":")) + "\n").encode("ascii")
                stream.write(gzip.compress(data, compresslevel=3, mtime=0))
                stream.flush()
                attempts += len(ids)
                timeouts += sum(c["phase"] == "timeout" for c in batch["records"])
                if i % 40 == 0 or i == 477:
                    print(f"Replicate {replicate}: {i}/477 queries, {attempts} attempts", flush=True)
        assert attempts == 94800
        audit.write_json(folder / f"cold{replicate}_receipt.json", {
            "attempts": attempts, "timeouts": timeouts, "cache_hits": 0,
            "collection_wall_seconds": time.perf_counter() - started,
            "archive_sha256": audit.sha_file(folder / f"cold{replicate}.jsonl.gz"),
        })


def read_rep(folder, plan, hashes, replicate):
    filename = folder / f"cold{replicate}.jsonl.gz"
    receipt = audit.read_json(folder / f"cold{replicate}_receipt.json")
    assert receipt["cache_hits"] == 0 and receipt["archive_sha256"] == audit.sha_file(filename)
    with gzip.open(filename, "rt", encoding="ascii") as stream:
        for (qid, row), line in zip(plan["rows"].items(), stream, strict=True):
            batch = json.loads(line)
            records = batch["records"]
            ids = row["top"] if row["example"] else []
            assert batch["qid"] == qid and [r["id"] for r in records] == ids
            assert batch["fingerprint"] == identity(plan, replicate, qid, ids, hashes)
            assert all(isinstance(r["passed"], bool) and (not r["passed"] or r["exit_code"] == 0)
                       and 0 <= r["completed_after_seconds"] <= batch["seconds"] for r in records)
            yield qid, batch


def project(row, batch):
    records = batch["records"]
    arms = depth.rank_row(row, records)
    original = {did: i for i, did in enumerate(row["top"], 1)}
    control = {did: i for i, did in enumerate(arms["K50"]["top200"], 1)}
    control_rank = arms["K50"]["rank"]
    for arm, result in arms.items():
        seen = records[:result["attempts"]]
        passing = {c["id"] for c in seen if c["passed"]}
        order = sorted(original, key=lambda did: (did not in passing, original[did]))
        assert order == result["top200"] and len(order) == 200
        assert result["rank"] == (order.index(row["gold"]) + 1 if row["gold"] in original else row["rank"])
        assert result["gold_position_among_passers"] == (result["rank"] if row["gold"] in passing else None)
        result["overtakers"] = [did for i, did in enumerate(result["passes"], 1)
                                if did != row["gold"] and i < result["rank"] and control[did] > control_rank]
        result["rank_change"] = control_rank - result["rank"]
        result["prefix_seconds"] = (batch["seconds"] if len(seen) == len(records) else
                                    max(c["completed_after_seconds"] for c in seen))
        result["timeouts"] = sum(c["phase"] == "timeout" for c in seen)
    return arms


def compare_records(qid, left, right, noise, key):
    comparison = noise.setdefault(key, {"pairs": 0, "verdict_differences": 0, "timeout_state_differences": 0,
                                         "verdict_differences_involving_timeout": 0, "changes": []})
    comparison["pairs"] += len(left)
    for a, b in zip(left, right, strict=True):
        assert a["id"] == b["id"]
        verdict_change = a["passed"] != b["passed"]
        timeout_change = (a["phase"] == "timeout") != (b["phase"] == "timeout")
        comparison["verdict_differences"] += verdict_change
        comparison["timeout_state_differences"] += timeout_change
        comparison["verdict_differences_involving_timeout"] += verdict_change and timeout_change
        if verdict_change or timeout_change:
            comparison["changes"].append({"query": qid, "candidate": a["id"],
                                          "left": [a["passed"], a["phase"]], "right": [b["passed"], b["phase"]]})


def summarize(rows, groups, replicate, samples):
    summary = {}
    for group, ids in groups.items():
        draws = depth.np.random.default_rng(audit.SEED).integers(0, len(ids), size=(10000, len(ids)))
        control = depth.np.array([depth.metric(rows[q]["K50"]["rank"])[0] for q in ids])
        stats = {}
        for arm in depth.ARMS:
            entries = [rows[q][arm] for q in ids]
            ndcg, mrr, recall = map(depth.np.asarray, zip(*(depth.metric(e["rank"]) for e in entries)))
            delta = ndcg - control
            boot = delta[draws].mean(axis=1) * 100
            samples[f"rep{replicate}_{group}_{arm}"] = boot
            attempts = sum(e["attempts"] for e in entries)
            passes = sum(len(e["passes"]) for e in entries)
            pos = Counter(e["gold_position_among_passers"] for e in entries if e["gold_position_among_passers"])
            ng = passes - pos.total()
            stats[arm] = {
                "queries": len(ids), "attempts": attempts, "additional_attempts": attempts - sum(rows[q]["K50"]["attempts"] for q in ids),
                "ndcg": float(ndcg.mean()*100), "mrr": float(mrr.mean()*100), "hits10": int(recall.sum()),
                "recall10": float(recall.mean()*100), "delta_ndcg": float(delta.mean()*100),
                "ci": depth.np.percentile(boot, [2.5, 97.5]).tolist(),
                "passes": passes, "passers_per_query": passes/len(ids), "zero_pass_queries": sum(not e["passes"] for e in entries),
                "gold_passes": pos.total(), "gold_positions_among_passers": dict(sorted(pos.items())),
                "non_gold_passes": ng, "non_gold_pct": 100*ng/attempts,
                "wins": sum(e["rank_change"] > 0 for e in entries), "losses": sum(e["rank_change"] < 0 for e in entries),
                "ties": sum(e["rank_change"] == 0 for e in entries),
                "queries_overtaken": sum(bool(e["overtakers"]) for e in entries),
                "gold_rank_median": statistics.median(e["rank"] for e in entries),
                "prefix_time": depth.latency_metrics([e["prefix_seconds"] for e in entries]),
                "timeouts": sum(e["timeouts"] for e in entries),
            }
        summary[group] = stats
    for arm in depth.ARMS[1:]:
        s = summary["pooled"][arm]
        s["gates"] = {"pooled_gain_and_ci": s["delta_ndcg"] >= 1 and s["ci"][0] > 0,
                      "both_halves_positive": min(summary[g][arm]["delta_ndcg"] for g in ("A", "B")) > 0,
                      "non_gold_rate": s["non_gold_passes"] * 200 <= s["attempts"]}
    return summary


def compressed_json(path, value):
    with path.open("xb") as f:
        f.write(gzip.compress(json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("ascii"), mtime=0))


def analyze(folder, destination):
    frozen = check_frozen()
    current, baseline, _, hashes = inputs(folder)
    plan = audit.read_json(folder / "plan.json")
    for key in plan.keys() - {"runtime"}:
        assert json.loads(json.dumps(current[key])) == plan[key]
    for key in ("verifier_sha256", "windows_job_sha256", "workers", "timeout"):
        assert current["runtime"][key] == plan["runtime"][key]
    controls = audit.read_json(V2 / "plan.json")
    groups = {**plan["halves"], "pooled": list(plan["rows"])}
    projected = {k: {} for k in REPS}
    noise, samples, summary, warm = {}, {}, {}, {}
    for first, second in zip(read_rep(folder, plan, hashes, 1), read_rep(folder, plan, hashes, 2), strict=True):
        q, batch1 = first
        assert q == second[0]
        row = plan["rows"][q]
        batch2 = second[1]
        compare_records(q, batch1["records"], batch2["records"], noise, "rep1_vs_rep2")
        ctrl_rows = []
        for ctrl in ("control1", "control2"):
            saved = audit.read_json(V2 / ctrl / f"{q}.json")
            ids = row["top"][:50] if row["example"] else []
            assert [c["id"] for c in saved["records"]] == ids
            assert saved["fingerprint"] == depth.cache_key(controls, q, ids, hashes, ctrl)
            ctrl_rows.append(saved["records"])
        for rep, batch in zip(REPS, (batch1, batch2), strict=True):
            projected[rep][q] = project(row, batch)
            for j, ctrl in enumerate(ctrl_rows, 1):
                compare_records(q, ctrl, batch["records"][:50], noise, f"control{j}_vs_rep{rep}")
    assert len(projected[1]) == len(projected[2]) == 477
    for rep in REPS:
        summary[rep] = summarize(projected[rep], groups, rep, samples)
        t = time.perf_counter()
        confirmed = 0
        for q, batch in read_rep(folder, plan, hashes, rep):
            assert project(plan["rows"][q], batch) == projected[rep][q]
            confirmed += len(batch["records"])
        assert confirmed == 94800
        warm[rep] = {"seconds": time.perf_counter() - t, "cached_verdicts": confirmed, "launches": 0}
    decision = {arm: all(all(summary[r]["pooled"][arm]["gates"].values()) for r in REPS) for arm in depth.ARMS[1:]}
    for entry in noise.values():
        entry["affected_verdict_queries"] = len({c["query"] for c in entry["changes"] if c["left"][0] != c["right"][0]})
    stage1 = {}
    for source, stats in baseline["source_metrics"].items():
        if source != depth.SOURCE:
            metrics = [depth.metric(r["rank"]) for r in baseline["rows"].values() if r["source"] == source]
            stage1[source] = {**stats, "queries": len(metrics), "mrr": statistics.mean(r[1] for r in metrics)*100}
    if destination != folder:
        destination.mkdir()
    for rep in REPS:
        compressed_json(destination / f"rankings{rep}.json.gz", projected[rep])
    with (destination / "bootstrap.npz").open("xb") as out:
        depth.np.savez_compressed(out, **samples)
    audit.write_json(destination / "noise.json", noise)
    audit.write_json(destination / "results.json", {"by_replicate": summary, "qualified": decision, "warm_replay": warm,
                                                   "stage1_other_sources": stage1, "frozen_after": frozen})
    print(json.dumps(decision))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["collect", "analyze"])
    parser.add_argument("--folder", type=Path, default=RUN)
    parser.add_argument("--output", type=Path, help="New analysis directory for a replay of saved raw data")
    args = parser.parse_args()
    if args.mode == "collect":
        if args.output:
            parser.error("--output applies only to analyze")
        collect(args.folder)
    else:
        analyze(args.folder, args.output or args.folder)
