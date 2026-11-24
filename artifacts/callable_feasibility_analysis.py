"""Summarize the callable experiment and run separate training-only controls."""
import argparse
from collections import Counter
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from callable_feasibility import (
    HERE, SOURCES, candidate_trial, compatibility, compute_rank, encode, ranking_summary,
    read_json, run_case, workload, write_json,
)
from callable_fixture_reference import REFERENCES


def rates(result, plan):
    rows = result["rows"]
    output = {}
    for kind in ("train_corpus", "saved_diagnostic"):
        scopes = {"overall": list(rows), **{source: result["populations"][source] for source in SOURCES}}
        output[kind] = {}
        for scope, ids in scopes.items():
            counts, structures, eligibility = Counter(), Counter(), Counter()
            for qid in ids:
                row = rows[qid]
                candidates = row["candidate_lists"][kind]["ids"]
                counts["top50_slots"] += len(candidates)
                if row["gold"] in candidates:
                    counts["gold_top50"] += 1
                structures.update(row["structures"].get(did, "outside_training_partition") for did in candidates)
                if qid not in plan["fixtures"]:
                    continue
                counts["fixture_queries"] += 1
                for did in candidates:
                    state = row["eligibility"].get(did, "outside_training_partition")
                    eligibility[state] += 1
                    cohort = "gold" if did == row["gold"] else "nongold"
                    counts[f"{cohort}_slots_on_fixture_queries"] += 1
                    counts[f"{cohort}_train_slots_on_fixture_queries"] += state != "outside_training_partition"
                    counts[f"{cohort}_compatible"] += state == "compatible"
                    counts[f"{cohort}_first_pass"] += did in row["one_case_passers"]
                    counts[f"{cohort}_all_pass"] += did in row["all_case_passers"]
                passes = set(row["all_case_passers"])
                row["candidate_lists"][kind]["signature_only_rank"] = compute_rank(
                    candidates, row["gold"], {did for did, state in row["eligibility"].items() if state == "compatible"}
                )
                assert passes.issubset(row["executions"])
            for qid in ids:
                entry = rows[qid]["candidate_lists"][kind]
                entry.setdefault("signature_only_rank", entry["baseline_rank"])
            output[kind][scope] = {"counts": dict(counts), "structures": dict(structures),
                                   "eligibility": dict(eligibility), "signature_only": ranking_summary(
                                       rows, ids, kind, "signature_only_rank")}
        for scope in ("reviewed", "fixture_eligible"):
            output[kind][scope] = {"signature_only": ranking_summary(
                rows, result["populations"][scope], kind, "signature_only_rank")}
    return output


def explicit_case(args, expected):
    return {"args": encode(args), "expected": encode(expected)}


def synthetic_checks(plan):
    fixture = {"owner": None, "name": "entry", "params": ["x"]}
    one = explicit_case((4,), 1)
    entries = {
        "typed_return": ("def entry(x): return 1", one, "pass"),
        "bool_is_not_integer": ("def entry(x): return True", one, "mismatch"),
        "list_is_not_tuple": ("def entry(x): return [x]", explicit_case((4,), (4,)), "mismatch"),
        "stdout_is_not_return": ("def entry(x): print(1)", one, "mismatch"),
        "successful_early_exit_has_no_return": ("def entry(x): print(1); raise SystemExit(0)", one, "invalid_return_record"),
        "nonzero_exit": ("def entry(x): print(1); raise SystemExit(7)", one, "nonzero"),
        "unsupported_output": ("def entry(x): return float(x)", one, "unsupported_return"),
        "state_fresh": ("n = 0\ndef entry(x):\n global n\n n += 1\n return n", one, "pass"),
        "cwd_and_environment": (
            "from pathlib import Path\nimport os\ndef entry(x):\n"
            " Path('local.txt').write_text('ok')\n"
            " return [str(Path.cwd()), Path('local.txt').read_text(), 'CTRLFIND_CALLABLE_SENTINEL' in os.environ]",
            one, "mismatch"
        ),
    }
    output = {}
    with patch.dict(os.environ, CTRLFIND_CALLABLE_SENTINEL="parent only"):
        for name, (source, case, expected) in entries.items():
            result = run_case(source, fixture, case)
            assert result["status"] == expected, (name, result)
            output[name] = result
    result = run_case(entries["state_fresh"][0], fixture, one)
    assert result["status"] == "pass"
    output["state_fresh_repeat"] = result
    temp_path, content, leaked = output["cwd_and_environment"]["return"][1]
    assert content == ["str", "ok"] and leaked == ["bool", False]
    assert not Path(temp_path[1]).exists()
    output["cwd_removed"] = True

    # Wait for a real descendant PID, then check termination through a Windows process handle.
    import _winapi
    with tempfile.TemporaryDirectory(prefix="callable_tree_control_") as directory:
        marker = Path(directory) / "child.pid"
        child = f"import os,time; from pathlib import Path; Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(10)"
        source = ("import subprocess,sys,time\ndef entry(x):\n"
                  f" subprocess.Popen([sys.executable, '-I', '-S', '-c', {child!r}])\n"
                  " time.sleep(10)\n return 1")
        result = run_case(source, fixture, one)
        assert result["status"] == "timeout" and marker.exists(), result
        pid = int(marker.read_text())
        try:
            handle = _winapi.OpenProcess(0x00100000, False, pid)
        except OSError as error:
            assert error.winerror == 87
        else:
            try:
                assert _winapi.WaitForSingleObject(handle, 0) == _winapi.WAIT_OBJECT_0
            finally:
                _winapi.CloseHandle(handle)
        output["timeout_descendant"] = {**result, "pid": pid, "terminated": True}

    # q452 publishes three True cases. A deliberately wrong constant can pass them all.
    fixture = plan["fixtures"]["q452"]
    source = "class Solution:\n    def isRationalEqual(self, S, T): return True"
    expected = REFERENCES["q452"]("0.(3)", "0.3")
    assert expected is False
    witness = explicit_case(("0.(3)", "0.3"), expected)
    trial, extra = candidate_trial(source, fixture), run_case(source, fixture, witness)
    assert all(row["status"] == "pass" for row in trial) and extra["status"] == "mismatch"
    output["constant_true_false_positive"] = {"published": trial, "witness_fixture": witness, "witness": extra}
    return output


def off_list_controls(result, plan, docs):
    output = {}
    for qid, fixture in plan["fixtures"].items():
        seen = result["rows"][qid]["eligibility"]
        compatible = [did for did in sorted(docs) if did not in seen and compatibility(docs[did], fixture) == "compatible"]
        if compatible:
            output[qid] = {did: candidate_trial(docs[did], fixture) for did in compatible}
    return output


def retrieval_provenance(fixed, ablation, docs, query_ids, audit):
    counts, gained = Counter(), []
    for qid in query_ids:
        old = fixed["rows"][qid]["stage1_top50"]
        new = ablation["rows"][qid]["top"]
        old_training = [did for did in old if did in docs]
        counts["filtered_prefix_same_set"] += set(old_training) == set(new[:len(old_training)])
        counts["filtered_prefix_same_order"] += old_training == new[:len(old_training)]
        did = fixed["rows"][qid]["gold_id"]
        counts["saved_hits"] += did in old
        counts["training_hits"] += did in new
        if did in new and did not in old:
            gained.append(qid)
            counts["gained_with_nontraining_slots"] += len(old_training) < 50
        counts["lost"] += did in old and did not in new
    q1694 = audit["rows"]["q1694"]
    assert q1694["source"] == "codeforces.com" and q1694["issue"] == "placeholder_or_truncation"
    raw = [qid for qid, row in audit["rows"].items() if row["source"] == "codeforces.com" and row["current"]]
    return {"counts": dict(counts), "gained_ids": gained, "codeforces_raw": raw,
            "codeforces_usable": sorted(set(raw) - {"q1694"}), "q1694_extracted": q1694["current"]}


def repeat_check(before, after):
    def decisions(dataset):
        return {(qid, did, index): (record["status"], record.get("return"), record.get("returncode"))
                for qid, row in dataset["rows"].items() for did, cases in row["executions"].items()
                for index, record in enumerate(cases)}

    a, b = decisions(before), decisions(after)
    assert a.keys() == b.keys()
    return {"case_verdicts": len(a), "differences": [key for key in a if a[key] != b[key]],
            "identical_ranking": before["ranking"] == after["ranking"], "timing": after["timing"]}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("input", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--repeat", type=Path)
    args = cli.parse_args()
    result = read_json(args.input)
    plan = read_json(HERE / "callable_verified_fixtures_train.json")
    _, docs, _, fixed, ablation, ids, _ = workload()
    audit = read_json(HERE / "source_example_audit_train.json")
    report = {"ranking": result["ranking"], "counts_and_signature_control": rates(result, plan),
              "provenance": retrieval_provenance(fixed, ablation, docs, ids, audit),
              "synthetic": synthetic_checks(plan), "off_list_train": off_list_controls(result, plan, docs)}
    if args.repeat:
        report["repeat"] = repeat_check(result, read_json(args.repeat))
    write_json(args.output, report)


if __name__ == "__main__":
    main()
