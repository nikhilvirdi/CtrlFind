"""Fresh Registration v2 controls. Use the v1 observation code, never its checkpoints."""
import argparse
import json
import os
import re
import statistics
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import verification_depth as depth


audit = depth.audit
HERE = audit.HERE
ROOT = HERE / "verification_depth_v2_run"
PHASES = ("control1", "control2")


def registration():
    text = (audit.ROOT / "docs/research-log.md").read_text(encoding="utf-8")
    header = "### Registration v2\n"
    assert text.count(header) == 1
    section = text.split(header)[1].split("\n### ", 1)[0].split("\n## ", 1)[0].strip()
    assert section.count("STOP CONDITIONS") == 1 and "independent full cold runs" in section
    assert "Frozen baseline integrity cannot be confirmed." in section
    return {"text": header + "\n" + section, "sha256": depth.digest(section)}


def fresh(folder, phase, plan, docs, hashes, limit):
    directory = folder / phase
    directory.mkdir()
    for index, (qid, row) in enumerate(plan["rows"].items(), 1):
        ids = row["top"][:limit] if row["example"] else []
        _, state, _ = depth.run_cached(directory / f"{qid}.json", plan, qid, ids, docs, hashes, phase)
        assert state == "miss"
        if index % 50 == 0 or index == 477:
            print(phase, index, "/ 477", flush=True)
    assert index == 477


def read_checked(folder, phase, plan, qid, hashes, limit):
    row = plan["rows"][qid]
    ids = row["top"][:limit] if row["example"] else []
    batch = audit.read_json(folder / phase / f"{qid}.json")
    assert batch["fingerprint"] == depth.cache_key(plan, qid, ids, hashes, phase)
    assert [r["id"] for r in batch["records"]] == ids
    assert all(isinstance(r["passed"], bool) and (not r["passed"] or r["phase"] == "exit_zero") for r in batch["records"])
    return batch


def controls(folder, plan, docs, hashes):
    assert not any((folder / phase).exists() for phase in (*PHASES, "shared", "noise.json"))
    for phase in PHASES:
        fresh(folder, phase, plan, docs, hashes, 50)
    compare_controls(folder, plan, hashes)


def compare_controls(folder, plan, hashes):
    archived = audit.read_json(audit.ALL_LEDGER)
    policy = archived["verification"]
    assert policy["k"] == 50 and policy["workers"] == plan["runtime"]["workers"]
    assert policy["timeout_seconds"] == plan["runtime"]["timeout"]
    for key in ("verifier_sha256", "windows_job_sha256"):
        assert policy[key] == plan["runtime"][key]
    for key in ("python", "platform", "utf8_mode"):
        assert archived["environment"][key] == plan["runtime"][key]
    history = archived["rows"]
    measurements = {}
    for qid, row in plan["rows"].items():
        batches = [read_checked(folder, phase, plan, qid, hashes, 50) for phase in PHASES]
        first, second = (b["records"] for b in batches)
        passers = [depth.passes(recs) for recs in (first, second)]
        changes = [{"id": a["id"], "first_pass": a["passed"], "second_pass": b["passed"],
                    "first_phase": a["phase"], "second_phase": b["phase"]}
                   for a, b in zip(first, second, strict=True)
                   if a["passed"] != b["passed"] or (a["phase"] == "timeout") != (b["phase"] == "timeout")]
        orders = [[*p, *(did for did in row["top"] if did not in p)] for p in passers]
        ranks = [order.index(row["gold"]) + 1 if row["gold"] in order else row["rank"] for order in orders]
        old = history.get(qid)
        comparable = bool(old and old["example_found"] and old["final_gold_rank"] is not None
                          and row["example"] and old["stage1_top50"] == row["top"][:50])
        measurements[qid] = {"attempts": len(first), "passes": passers,
                             "phase_counts": [dict(Counter(r["phase"] for r in recs)) for recs in (first, second)],
                             "seconds": [b["seconds"] for b in batches], "rankings_equal": orders[0] == orders[1],
                             "ranks": ranks, "changes": changes,
                             "historical_passers": old["passer_ids"] if comparable else None}
    groups = {**plan["halves"], "pooled": list(plan["rows"])}
    summaries = {}
    for group, ids in groups.items():
        rows = [measurements[q] for q in ids]
        matched = [r for r in rows if r["historical_passers"] is not None]
        old_count = sum(len(r["historical_passers"]) for r in matched)
        current = [sum(len(r["passes"][i]) for r in matched) for i in range(2)]
        disagreements = [x for r in rows for x in r["changes"] if x["first_pass"] != x["second_pass"]]
        summaries[group] = {
            "queries": len(ids), "candidate_verdict_pairs": sum(r["attempts"] for r in rows),
            "passing_counts": [sum(len(r["passes"][i]) for r in rows) for i in range(2)],
            "verdict_disagreements": len(disagreements),
            "queries_with_verdict_disagreements": sum(r["passes"][0] != r["passes"][1] for r in rows),
            "timeout_verdict_disagreements": sum("timeout" in (x["first_phase"], x["second_phase"]) for x in disagreements),
            "timeouts": [sum(r["phase_counts"][i].get("timeout", 0) for r in rows) for i in range(2)],
            "timeout_state_disagreements": sum((x["first_phase"] == "timeout") != (x["second_phase"] == "timeout")
                                               for r in rows for x in r["changes"]),
            "rankings_differ": sum(not r["rankings_equal"] for r in rows),
            "gold_ranks_differ": sum(r["ranks"][0] != r["ranks"][1] for r in rows),
            "ndcg_mrr_recall": [[100 * statistics.mean(depth.metric(r["ranks"][i])[j] for r in rows)
                                 for j in range(3)] for i in range(2)],
            "cold_seconds": [depth.latency_metrics([r["seconds"][i] for r in rows]) for i in range(2)],
            "historical_matched_queries": len(matched), "historical_passes": old_count,
            "matched_control_passes": current, "matched_count_noise": abs(current[1] - current[0]),
            "historical_disagreements": [sum(len(set(r["historical_passers"]) ^ set(r["passes"][i])) for r in matched)
                                         for i in range(2)]}
    pooled = summaries["pooled"]
    assert pooled["candidate_verdict_pairs"] == 474 * 50
    consistent = bool(pooled["historical_matched_queries"]) and all(
        abs(c - pooled["historical_passes"]) <= pooled["matched_count_noise"] for c in pooled["matched_control_passes"])
    output = {"groups": summaries, "rows": measurements, "historical_counts_within_noise": consistent}
    noise_path = folder / "noise.json"
    if noise_path.exists():
        assert audit.read_json(noise_path) == output
    else:
        audit.write_json(noise_path, output)
    print(json.dumps(summaries["pooled"]), flush=True)
    if not consistent:
        raise SystemExit("STOP: historical passing counts outside the measured noise floor or no comparable rows")


def sweep(folder, plan, docs, hashes):
    noise = audit.read_json(folder / "noise.json")
    signoff = audit.read_json(folder / "control_validation.json")
    assert noise["historical_counts_within_noise"] and signoff["continue"]
    assert signoff["noise_sha256"] == audit.sha_file(folder / "noise.json")
    controls_plan = audit.read_json(folder / "plan.json")
    shared = folder / "shared"
    direct = folder / "direct"
    shared.mkdir()
    direct.mkdir()
    timings = {}
    for n, (qid, row) in enumerate(plan["rows"].items(), 1):
        ids = row["top"] if row["example"] else []
        first = read_checked(folder, PHASES[0], controls_plan, qid, hashes, 50)
        second = read_checked(folder, PHASES[1], controls_plan, qid, hashes, 50)
        observations = {"K50": [first]}
        order = ("K200", "K100", "ADAPT")
        off = (n - 1) % len(order)
        order = order[off:] + order[:off]
        for arm in order:
            if arm == "K200":
                record, state, _ = depth.run_cached(shared / f"{qid}.json", plan, qid, ids, docs, hashes, "shared_K200")
            else:
                end = 50 if arm == "ADAPT" else 100
                record, state, _ = depth.run_cached(direct / f"{qid}_{arm}.json", plan, qid, ids[:end], docs, hashes, arm)
            assert state == "miss"
            observations[arm] = [record]
            if arm == "ADAPT" and ids and not depth.passes(record["records"]):
                extension, state, _ = depth.run_cached(direct / f"{qid}_extension.json", plan, qid, ids[50:],
                                                        docs, hashes, "ADAPT_extension")
                assert state == "miss"
                observations[arm].append(extension)
        ref = observations["K200"][0]["records"]
        comparison = {**observations, "K50_repeat": [second]}
        times = {}
        discrepancies = []
        for arm, batches in comparison.items():
            fresh_records = [r for batch in batches for r in batch["records"]]
            assert [r["id"] for r in fresh_records] == ids[:len(fresh_records)]
            for cand, census in zip(fresh_records, ref):
                if cand["passed"] != census["passed"]:
                    discrepancies.append({"arm": arm, "id": cand["id"], "arm_record": cand, "census_record": census})
            times[arm] = {"seconds": sum(batch["seconds"] for batch in batches), "attempts": len(fresh_records),
                          "batch_seconds": [batch["seconds"] for batch in batches], "cache_hits": 0,
                          "verdict_differences": [d for d in discrepancies if d["arm"] == arm],
                          "phase_counts": dict(Counter(r["phase"] for r in fresh_records))}
        timings[qid] = {"half": row["half"], "order_after_controls": order, "arms": times}
        if discrepancies:
            audit.write_json(folder / "inconsistency.json", {"qid": qid, "details": discrepancies, "times_so_far": timings})
            raise SystemExit(f"STOP: observed arm verdict differences for {qid}. No later query was run.")
        if n % 25 == 0 or n == 477:
            print("sweep", n, "/ 477", flush=True)
    audit.write_json(folder / "timing.json", timings)


def finish(folder, plan, baseline, hashes):
    """Require a review ledger. Retain the exact paired bootstrap distribution for every arm."""
    evidence_path = folder / "evidence.json"
    evidence = audit.read_json(evidence_path)["pairs"]
    results = depth.finish(folder, plan, baseline, hashes, evidence_path)
    for qid, row in results["rows"].items():
        candidates = set(row["arms"]["K200"]["passes"]) - {row["gold"]}
        assert set(evidence.get(qid, {})) == candidates
        for entry in evidence.get(qid, {}).values():
            assert entry["status"] in ("false", "correct", "unknown") and entry["reason"]
            assert entry["status"] == "unknown" or entry["evidence"]
    samples = {}
    for group, ids in {**plan["halves"], "pooled": list(plan["rows"])}.items():
        draws = depth.np.random.default_rng(audit.SEED).integers(0, len(ids), size=(10_000, len(ids)))
        for arm in depth.ARMS:
            diff = depth.np.array([depth.metric(results["rows"][q]["arms"][arm]["rank"])[0]
                                   - depth.metric(results["rows"][q]["arms"]["K50"]["rank"])[0] for q in ids])
            samples[f"{group}_{arm}"] = diff[draws].mean(axis=1) * 100
            expected = results["summaries"][group][arm]["ndcg_delta_ci_95"]
            assert depth.np.allclose(depth.np.quantile(samples[f"{group}_{arm}"], [0.025, 0.975]), expected)
    with (folder / "bootstrap.npz").open("xb") as f:
        depth.np.savez_compressed(f, **samples)
    audit.write_json(folder / "results.json", results)
    if any(r["safety"] == "inconclusive" for r in results["acceptance"].values()):
        raise SystemExit("STOP: unresolved passes straddle the safety threshold")


def shell_identity():
    # The parent shell assigns a fresh receipt and a disposable directory to every command.
    # Keep both as provenance. No process receives another process's environment.
    return {"receipt": os.environ["CODEX_TOOL_CALL_ID"],
            "scratch_id": re.search(r"#!sandbox\.\{([\da-f-]+)\}", os.environ["TEMP"], re.I)[1]}


def check_phase_inputs(current, archived, identity):
    saved_env = dict(os.environ)
    current_id = shell_identity()["scratch_id"]
    for key in ("LOCALAPPDATA", "TEMP", "TMP", "TORCHINDUCTOR_CACHE_DIR"):
        assert current_id in saved_env[key]
        saved_env[key] = saved_env[key].replace(current_id, identity["scratch_id"])
    saved_env["CODEX_TOOL_CALL_ID"] = identity["receipt"]
    replay = json.loads(json.dumps(current))
    replay["runtime"]["inherited_environment_sha256"] = audit.json_digest(sorted(saved_env.items()))
    assert replay == archived, "An input, execution setting, or non-ephemeral environment value changed"


def setup(folder, phase):
    audit.configure()
    assert audit.torch.get_num_threads() == 4 and audit.torch.get_num_interop_threads() == 1
    folder.mkdir(exist_ok=True)
    with patch.object(depth, "registration", registration):
        plan, baseline, docs, hashes = depth.prepare(folder, persist=False)
    control_path = folder / "plan.json"
    if phase == "controls":
        audit.write_json(control_path, plan)
        audit.write_json(folder / "control_shell.json", shell_identity())
    else:
        archived = audit.read_json(control_path)
        check_phase_inputs(plan, archived, audit.read_json(folder / "control_shell.json"))
        if phase == "compare-controls":
            plan = archived
        elif phase == "sweep":
            audit.write_json(folder / "sweep_plan.json", plan)
            audit.write_json(folder / "sweep_shell.json", shell_identity())
        else:
            archived = audit.read_json(folder / "sweep_plan.json")
            check_phase_inputs(plan, archived, audit.read_json(folder / "sweep_shell.json"))
            plan = archived
    return plan, baseline, docs, hashes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["controls", "compare-controls", "sweep", "finish"])
    parser.add_argument("--run-name", default=ROOT.name,
                        help="Fresh output directory beginning verification_depth_v2_run, under artifacts/")
    args = parser.parse_args()
    if not args.run_name.startswith(ROOT.name) or Path(args.run_name).name != args.run_name:
        parser.error("use a v2 experiment directory name")
    folder = HERE / args.run_name
    plan, baseline, docs, hashes = setup(folder, args.phase)
    if args.phase == "controls":
        controls(folder, plan, docs, hashes)
    elif args.phase == "compare-controls":
        compare_controls(folder, plan, hashes)
    elif args.phase == "sweep":
        sweep(folder, plan, docs, hashes)
    else:
        finish(folder, plan, baseline, hashes)
