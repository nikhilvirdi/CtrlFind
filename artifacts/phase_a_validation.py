"""Replay archived verdicts and retest only the two aborted Step 5 training queries."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import local
from unittest.mock import patch

import step5_adapt_validation as step5
import verification_depth_v2 as v2
from src import execute


audit, depth, v3 = step5.audit, step5.depth, step5.v3
Outcome = execute.VerificationOutcome
LIVE = {"q2346": "d978", "q1696": "d2279"}


def integrity():
    evidence = step5.evidence()
    step4 = audit.HERE / "step4_version_run"
    hashed = audit.read_json(step4 / "replay.json")["sha256"]
    for name, expected in hashed.items():
        assert audit.sha_file(step4 / name) == expected, name
    release = audit.read_json(audit.HERE / "step5_adapt_run" / "release_checks.json")
    for name, entry in release["new_output_inventory"].items():
        path = audit.ROOT / name
        assert path.stat().st_size == entry["bytes"] and audit.sha_file(path) == entry["sha256"], name
    return {"frozen_and_v3": evidence, "step4": hashed, "step5": release["new_output_inventory"]}


def replay(plan, queries, docs, hashes):
    ctrl = audit.read_json(v2.ROOT / "plan.json")
    assert plan["rows"] == ctrl["rows"]
    results = {}
    live_references = {}
    with TemporaryDirectory(prefix="ctrlfind_phase_a_replay_") as tmp, patch.object(
        execute, "_run_candidate", side_effect=AssertionError("Archive replay attempted execution")
    ):
        for rep, phase in zip(v3.REPS, v2.PHASES, strict=True):
            with gzip.open(v3.RUN / f"rankings{rep}.json.gz", "rt", encoding="ascii") as stream:
                rankings = json.load(stream)
            results[rep] = {"K50": 0, "ADAPT": 0, "queries": 0, "timeout_state_changes": 0}
            runners = {arm: step5.runner(Path(tmp), arm, docs) for arm in ("K50", "ADAPT")}
            for qid, batch in v3.read_rep(v3.RUN, plan, hashes, rep):
                row = plan["rows"][qid]
                control = v2.read_checked(v2.ROOT, phase, ctrl, qid, hashes, 50)["records"]
                for left, right in zip(control, batch["records"][:50], strict=True):
                    assert (left["id"], left["passed"]) == (right["id"], right["passed"]), qid
                    results[rep]["timeout_state_changes"] += (left["phase"] == "timeout") != (right["phase"] == "timeout")
                for arm, runner in runners.items():
                    expected = rankings[qid][arm]
                    records = control if arm == "K50" else batch["records"][:expected["attempts"]]
                    called = []

                    def stored(codes, example, timeout, *, workers, allow_trailing_padding, detailed):
                        ids = [candidate.did for candidate in codes]
                        expected_batch = records[len(called):len(called) + len(ids)]
                        assert ids == [entry["id"] for entry in expected_batch], (rep, qid, arm)
                        assert list(example) == row["example"] and timeout == 1 and detailed
                        assert workers == depth.WORKERS and allow_trailing_padding == row["padding"]
                        called.extend(ids)
                        return [Outcome.PASS if entry["passed"] else
                                Outcome.TIMEOUT if entry["phase"] == "timeout" else Outcome.FAIL
                                for entry in expected_batch]

                    with patch("src.mteb_wrappers.verify", side_effect=stored):
                        order = step5.search(runner, qid, row, queries)
                    assert order == expected["top200"], (rep, qid, arm)
                    assert called == [entry["id"] for entry in records], (rep, qid, arm)
                    assert [entry["id"] for entry in records if entry["passed"]] == expected["passes"]
                    with patch("src.mteb_wrappers.verify", side_effect=AssertionError("Replay cache miss")):
                        assert step5.search(runner, qid, row, queries) == order
                    results[rep][arm] += len(called)
                results[rep]["queries"] += 1
                if rep == 1 and qid in LIVE:
                    live_references[qid] = rankings[qid]["ADAPT"]
            assert results[rep]["K50"] == 23_700 and results[rep]["ADAPT"] == 79_650
            for runner in runners.values():
                runner.checkpoint.unlink()
    return results, live_references


def retest(plan, queries, docs, references):
    results = {}
    verify = execute.verify
    with TemporaryDirectory(prefix="ctrlfind_phase_a_live_") as tmp:
        runner = step5.runner(Path(tmp), "ADAPT", docs)
        for qid, incident in LIVE.items():
            row = plan["rows"][qid]
            batches, records = [], {}
            launch = execute._run_candidate
            current = local()
            capture = execute._capture_snippet

            def instrument(code, *args):
                current.did = code.did
                return capture(code, *args)

            def observe(*args, **kwargs):
                try:
                    return launch(*args, **kwargs)
                except execute._windows_job.PipeCaptureError as exc:
                    records[current.did] = str(exc)
                    raise

            def checks(codes, *args, **kwargs):
                outcomes = verify(codes, *args, **kwargs)
                batches.append(dict(zip([c.did for c in codes], outcomes, strict=True)))
                return outcomes

            with (patch.object(execute, "_run_candidate", side_effect=observe),
                  patch.object(execute, "_capture_snippet", side_effect=instrument),
                  patch("src.mteb_wrappers.verify", side_effect=checks)):
                order = step5.search(runner, qid, row, queries)
            measured = {did: outcome for batch in batches for did, outcome in batch.items()}
            passed = [did for did, outcome in measured.items() if outcome is Outcome.PASS]
            cached = runner.checkpoint.exists() and bool(runner.checkpoint.read_bytes().strip())
            assert measured[incident] is Outcome.INFRASTRUCTURE_ERROR and incident not in passed
            assert incident in records and not cached
            reference = references[qid]
            results[qid] = {"candidate": incident, "outcomes": dict(Counter(measured.values())),
                            "launches": len(measured), "batch_sizes": list(map(len, batches)),
                            "passers": passed, "archived_passers": reference["passes"],
                            "rank_order_matches_archive": order == reference["top200"],
                            "capture_errors": records, "checkpoint_written": cached,
                            "candidate_outcomes": measured, "order": order}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not __debug__ or args.output.exists():
        parser.error("Use a new output path and keep assertions enabled")
    before = integrity()
    plan, runtime, queries, docs, hashes = step5.inputs()
    archived, refs = replay(plan, queries, docs, hashes)
    print(f"Archive replay matched: {archived}", flush=True)
    live = retest(plan, queries, docs, refs) if args.live else {}
    assert integrity() == before
    audit.write_json(args.output, {"integrity": before, "runtime": runtime, "archive": archived,
                                  "archive_candidate_launches": 0, "live_training_queries": live})


if __name__ == "__main__":
    main()
