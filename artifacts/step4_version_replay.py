"""Check saved Step 4 manifests, rankings, verdict accounting and bootstrap without running candidates."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

import step4_version_retrieval as step4


audit, np = step4.audit, step4.np


def replay(folder):
    plan, results = (audit.read_json(folder / f"{name}.json") for name in ("plan", "results"))
    assert results["plan_sha256"] == audit.sha_file(folder / "plan.json")
    frozen, hashes = step4.integrity()
    assert plan["frozen"] == frozen and plan["production_source_sha256"] == hashes
    base, queries, gold, qids = step4.inputs()
    versions, rows = step4.manifest_versions(base, queries, gold, qids)
    assert json.loads(json.dumps(rows)) == plan["rows"]
    for name, docs in versions.items():
        assert {did: step4.content_hash(code) for did, code in docs.items()} == plan["manifests"][name]
        assert {did: code for did, code in docs.items() if code != base[did]} == plan["text_changes"][name]

    scoped = audit.read_json(folder / "scoped.json")
    scoped_passes = True
    for name, cases in scoped.items():
        assert [row["query"] for row in cases] == plan["scoped_query_ids"]
        for row in cases:
            hits = row["hits"]
            scoped_passes &= len({did for did, _ in hits}) == len(hits) == 50
            scoped_passes &= all(plan["manifests"][name].get(did) == digest for did, digest in hits)
            assert row["rank"] == next((i for i, (did, _) in enumerate(hits, 1) if did == gold[row["query"]]), None)

    labels = audit.read_json(folder / "mutant_verdicts.json")
    assert set(labels["labels"]) == {q for q in qids if rows[q]["status"] == "retained"}
    for q, item in labels["labels"].items():
        assert item["status"] == ("SURVIVOR" if item["versions"]["v2"]["passed"] else "KILLED"), q
    primary = [q for q in qids if q in labels["labels"] and labels["labels"][q]["status"] == "KILLED"
               and labels["labels"][q]["versions"]["v1"]["passed"]]
    saved = audit.read_json(folder / "bonus.json")
    assert primary == labels["primary"] == list(saved["rows"]) and len(primary) == results["primary"]
    catalog = saved["catalog"]
    assert len(catalog) == len({(e["doc_id"], e["content"]) for e in catalog}) == results["version_entries"]
    for name, manifest in plan["manifests"].items():
        assert {e["doc_id"]: e["content"] for e in catalog if name in e["versions"]} == manifest
    checked_slots = Counter()
    changes = Counter()
    for q, row in saved["rows"].items():
        original, reranked = row["similarity"], row["execution"]
        assert set(row["verdicts"]) == {str(i) for i in row["top50"]} and len(row["verdicts"]) == 50
        promoted = [i for i in row["top50"] if row["verdicts"][str(i)]["passed"]]
        promote_set = set(promoted)
        assert row["reranked_top50"] == promoted + [i for i in row["top50"] if i not in promote_set]
        positions = {rank for rank, i in enumerate(row["top50"], 1) if i in promote_set}

        def move(rank):
            count_before = sum(pos < rank for pos in positions)
            return count_before + 1 if rank in positions else rank + len(positions) - count_before

        assert reranked["passing_ranks"] == sorted(move(rank) for rank in original["passing_ranks"])
        assert reranked["first_failing_rank"] == move(original["first_failing_rank"])
        for outcome in (original, reranked):
            rank = outcome["passing_ranks"][0]
            assert outcome["passing_rank"] == rank
            assert outcome["working_first"] == (rank < outcome["first_failing_rank"])
            assert outcome["mrr"] == (1 / rank if rank <= 10 else 0)
            gain = sum(1 / math.log2(r + 1) for r in outcome["passing_ranks"] if r <= 10)
            ideal = sum(1 / math.log2(r + 1) for r in range(1, min(10, len(outcome["passing_ranks"])) + 1))
            assert outcome["ndcg"] == gain / ideal
        checked_slots.update(r["phase"] for r in row["verdicts"].values())
        changes["non_gold"] += sum(r["passed"] and catalog[int(i)]["doc_id"] != gold[q]
                                   for i, r in row["verdicts"].items())
        changes["siblings"] += sum(catalog[i]["doc_id"] == gold[q] for i in row["top50"])
    assert dict(checked_slots) == results["verification"]["slot_phases"]
    assert changes["non_gold"] == results["verification"]["non_gold_passes"]
    assert changes["siblings"] == results["verification"]["sibling_slots"]
    metrics, samples = step4.bootstrap(saved["rows"])
    assert metrics == results["quality"]
    with np.load(folder / "bootstrap.npz", allow_pickle=False) as original:
        assert set(original.files) == set(samples)
        assert all(np.array_equal(sample, original[name]) for name, sample in samples.items())
    structure, timings = (audit.read_json(folder / f"{name}.json") for name in ("structure", "timings"))
    assert all(len(pairs) == 3 for pairs in timings.values())
    gates = {
        "p1_structure": structure["ordered_ids_and_content_match"] and structure["allclose"],
        "p1_timing10": statistics.median(p["incremental"]["seconds"] for p in timings["10"]) <
                       statistics.median(p["full"]["seconds"] for p in timings["10"]),
        "p1_scope": scoped_passes, "bonus_population": len(primary) >= 100,
        "bonus_working_first": len(primary) >= 100 and metrics["working_first"]["delta"]["ci95"][0] > 0,
    }
    assert results["gates"] == {name: "PASSED" if ok else "FAILED" for name, ok in gates.items()}
    return {"gates": results["gates"], "reproduced_mutations": len(labels["labels"]),
            "replayed_bonus_queries": len(primary), "candidate_executions": 0,
            "bootstrap_distributions_identical": len(samples),
            "sha256": {p.name: audit.sha_file(p) for p in sorted(folder.iterdir()) if p.name != "replay.json"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    assert __debug__ and not (args.folder / "replay.json").exists()
    result = replay(args.folder)
    audit.write_json(args.folder / "replay.json", result)
    print(result)
