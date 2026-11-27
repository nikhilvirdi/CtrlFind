"""Validate the Windows cleanup change using the archived training controls."""
import argparse
from collections import Counter
import os
from pathlib import Path
import random
import re
import subprocess
import sys

import verification_depth_v2 as v2
import verification_depth_v3 as v3


audit = v2.audit
depth = v2.depth


def run_regressions():
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    command = [sys.executable, "-B", "-X", "utf8", "-m", "unittest", "discover", "-s", "tests", "-v"]
    result = subprocess.run(command, cwd=audit.ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    print(result.stderr, end="", flush=True)
    result.check_returncode()
    # The new Windows cases add two tests to the 39-case archived suite.
    summary = re.search(r"^Ran (\d+) tests in ([\d.]+)s\s+OK\s*\Z", result.stderr, re.M)
    if summary is None or int(summary[1]) != 41:
        raise ValueError("Expected a successful unittest footer for 41 tests")
    return {"command": command[1:], "exit_code": result.returncode, "count": int(summary[1]),
            "seconds": float(summary[2]), "outcome": "OK"}


def compare(reference, observed):
    totals = Counter()
    changes = {}
    assert reference.keys() == observed.keys()
    for qid, expected in reference.items():
        for old, new in zip(expected, observed[qid], strict=True):
            assert old["id"] == new["id"]
            before, after = old["phase"], new["phase"]
            outcome_changed = old["passed"] != new["passed"]
            timeout_changed = (before == "timeout") != (after == "timeout")
            totals["attempts"] += 1
            totals["reference_passes"] += old["passed"]
            totals["observed_passes"] += new["passed"]
            totals["reference_timeouts"] += before == "timeout"
            totals["observed_timeouts"] += after == "timeout"
            totals["verdict_changes"] += outcome_changed
            totals["timeout_status_changes"] += timeout_changed
            if outcome_changed or timeout_changed:
                changes.setdefault(qid, []).append({
                    "id": old["id"], "reference": [old["passed"], before],
                    "observed": [new["passed"], after],
                })
    return {"counts": dict(totals), "changes": changes}


def validate():
    frozen = v3.check_frozen()
    regression = run_regressions()
    archived = audit.read_json(v2.ROOT / "plan.json")
    v3_plan = audit.read_json(v3.RUN / "plan.json")
    audit.configure()
    current, _, docs, hashes = depth.prepare(v2.ROOT, persist=False)
    assert json_rows(current) == archived["rows"] == v3_plan["rows"]
    for key in ("python", "platform", "timeout", "workers", "encoding", "utf8_mode", "verifier_sha256"):
        assert current["runtime"][key] == archived["runtime"][key], key
    assert current["runtime"]["windows_job_sha256"] != archived["runtime"]["windows_job_sha256"]

    assert (v2.ROOT / "noise.json").is_file()
    v2.compare_controls(v2.ROOT, archived, hashes)
    controls = {phase: {qid: v2.read_checked(v2.ROOT, phase, archived, qid, hashes, 50)["records"]
                        for qid in archived["rows"]} for phase in v2.PHASES}
    replay = {}
    for i, phase in enumerate(v2.PHASES, 1):
        prefix = {qid: batch["records"][:50] for qid, batch in v3.read_rep(v3.RUN, v3_plan, hashes, i)}
        result = compare(controls[phase], prefix)
        assert result["counts"]["attempts"] == 23_700 and result["counts"]["verdict_changes"] == 0
        replay[phase] = result
    paired_controls = compare(controls["control1"], controls["control2"])
    assert paired_controls["counts"]["verdict_changes"] == 0
    del prefix

    rng = random.Random(audit.SEED)
    selected = {half: rng.sample(sorted(q for q in archived["halves"][half] if archived["rows"][q]["example"]), 100)
                for half in ("A", "B")}
    live = {}
    for index, qid in enumerate(selected["A"] + selected["B"], 1):
        row = current["rows"][qid]
        observed = depth.observe(row["top"][:50], docs, row)
        records = observed["records"]
        assert [record["id"] for record in records] == row["top"][:50]
        live[qid] = [{key: record[key] for key in ("id", "passed", "phase", "error") if key in record}
                     for record in records]
        if index % 25 == 0:
            print(f"Live K50: {index}/200 training queries", flush=True)
    live_comparisons = {phase: compare({qid: reference[qid] for qid in live}, live)
                        for phase, reference in controls.items()}
    assert all(result["counts"]["attempts"] == 10_000 for result in live_comparisons.values())
    assert v3.check_frozen() == frozen
    return {
        "training_revision": archived["dataset_revision"], "runtime": current["runtime"], "seed": audit.SEED,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=audit.ROOT, text=True).strip(),
        "regressions": regression, "archives_vs_v3_prefixes": replay, "archives_vs_each_other": paired_controls,
        "live_query_ids": selected, "live_vs_controls": live_comparisons, "live_verdicts": live,
        "frozen_reference": frozen,
    }


def json_rows(plan):
    return {qid: dict(row, example=list(row["example"]) if row["example"] else None)
            for qid, row in plan["rows"].items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not __debug__ or args.output.exists():
        parser.error("Use a new output path and keep assertions enabled")
    result = validate()
    audit.write_json(args.output, result)
    for phase, comparison in result["live_vs_controls"].items():
        print(phase, comparison["counts"])
