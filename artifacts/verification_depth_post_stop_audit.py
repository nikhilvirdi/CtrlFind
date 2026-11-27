"""Audit the saved Registration v3 run after the closing script stopped."""
import argparse
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import verification_depth_v3 as v3


EXPECTED_TESTS = 39


def read_unittest_footer(output):
    """Use unittest's suite count and outcome, which follow verbose test output."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) < 3 or not re.fullmatch(r"-{10,}", lines[-3]):
        raise ValueError("Missing unittest suite footer")
    summary = re.fullmatch(r"Ran (\d+) tests? in (\d+(?:\.\d+)?)s", lines[-2])
    if summary is None or int(summary[1]) != EXPECTED_TESTS or lines[-1] != "OK":
        raise ValueError(f"Expected {EXPECTED_TESTS} completed tests and an unqualified OK")
    return {"count": int(summary[1]), "seconds": float(summary[2]), "outcome": lines[-1]}


def validate():
    if not __debug__:
        raise RuntimeError("Run this audit with assertions enabled")
    audit = v3.audit
    folder = v3.RUN
    originals = {path.name: audit.sha_file(path) for path in sorted(folder.iterdir())}
    frozen = v3.check_frozen()
    if audit.read_json(folder / "frozen_before.json") != frozen:
        raise ValueError("The v3 frozen manifest differs from the retained v2 reference")
    regression = read_unittest_footer((folder / "regression_tests.txt").read_text(encoding="utf-8"))
    log = audit.ROOT / "docs" / "research-log.md"
    stop_heading = "### Closing validation stopped\n"
    stop_record = log.read_text(encoding="utf-8").split(stop_heading, 1)[1].split("\n### ", 1)[0].strip()
    assert "formal Step 3 closure is pending" in stop_record
    with TemporaryDirectory(prefix="ctrlfind-v3-post-stop-") as scratch:
        replay = Path(scratch) / "replay"
        with patch.object(v3.depth, "observe", side_effect=AssertionError("Candidate execution during audit")), \
                patch.object(v3.depth.execute, "verify", side_effect=AssertionError("Verification during audit")):
            v3.analyze(folder, replay)

        identical = {}
        for name in ("rankings1.json.gz", "rankings2.json.gz", "bootstrap.npz", "noise.json"):
            assert audit.sha_file(replay / name) == originals[name], name
            identical[name] = originals[name]
        saved = audit.read_json(folder / "results.json")
        replayed = audit.read_json(replay / "results.json")
        for result in (saved, replayed):
            for record in result["warm_replay"].values():
                record.pop("seconds")
        assert saved == replayed, "Replayed results differ after removing elapsed replay time"

    replicate_counts = {}
    for rep in v3.REPS:
        aggregate = replayed["by_replicate"][str(rep)]["pooled"]["K200"]
        receipt = audit.read_json(folder / f"cold{rep}_receipt.json")
        assert receipt["attempts"] == aggregate["attempts"] == 94_800
        assert receipt["timeouts"] == aggregate["timeouts"] and receipt["cache_hits"] == 0
        replicate_counts[str(rep)] = {key: receipt[key] for key in ("attempts", "timeouts", "cache_hits")}
    noise = audit.read_json(folder / "noise.json")["rep1_vs_rep2"]
    assert noise["pairs"] == 94_800 and noise["verdict_differences"] == 0
    assert v3.check_frozen() == frozen
    assert originals == {path.name: audit.sha_file(path) for path in sorted(folder.iterdir())}
    return {
        "kind": "post-stop audit",
        "original_protocol_status": "Stopped on the closing reporting-script error",
        "registration_sha256": v3.protocol()["sha256"],
        "stop_record_normalized_sha256": v3.depth.digest(stop_record),
        "archived_artifact_sha256": originals,
        "archived_unittest_footer": regression,
        "cold_replicates": replicate_counts,
        "verdict_pairs": noise["pairs"],
        "verdict_differences": noise["verdict_differences"],
        "timeout_state_differences": noise["timeout_state_differences"],
        "qualified": replayed["qualified"],
        "byte_identical_replay_sha256": identical,
        "results_match_except_elapsed_replay_time": True,
        "audit_candidate_launches": 0,
        "frozen_reference": frozen,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True, help="New JSON file outside the archived run")
    args = parser.parse_args()
    if args.receipt.resolve().is_relative_to(v3.RUN.resolve()) or args.receipt.exists():
        parser.error("Use a new receipt path outside the archived run")
    v3.audit.write_json(args.receipt, validate())
