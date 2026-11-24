"""Check the dev-stdin census against stage-1 MTEB and both archived 500-query runs."""
import argparse
from collections import Counter
from pathlib import Path

import retrieval_consistency as audit
from dev_stdin_audit import training_documents
from src.cache import text_key
from src.config import EMB_CACHE, MAX_SEQ_LEN, MODEL_NAME, QUERY_PREFIX
from src.execute import EXAMPLE_RE


def check_dev(measurements, repeated, vectors):
    from mteb.similarity_functions import cos_sim

    assert audit.sha_file(measurements) == audit.sha_file(repeated)
    result = audit.read_json(measurements)
    queries, sources, gold, sample, dev, old, ablation = audit.inputs()
    assert result["membership"]["query_ids"] == dev
    ids, _, formatted, _, passage_vectors, cache = training_documents(old, ablation)
    assert ids == result["corpus"]["document_ids"]
    query_vectors, manifest = audit.read_vectors(vectors / "dev_vectors.npz", dev, queries)
    assert manifest == result["query_vector_manifest"]
    order = audit.production_orders(dev, [queries[qid] for qid in dev], query_vectors,
                                   ids, formatted, passage_vectors, old["model"]["search_depth"])
    rows = result["rows"]
    production = {qid: {"rank": order[qid].index(gold[qid]) + 1 if gold[qid] in order[qid] else None}
                  for qid in dev}
    order_changes = []
    cutoff_ties, tied_gold_excluded = Counter(), Counter()
    historical_pattern_only = []
    scores = cos_sim(query_vectors, passage_vectors).numpy()
    positions = {did: i for i, did in enumerate(ids)}
    same_members = 0
    for qid, score_row in zip(dev, scores, strict=True):
        row, top200 = rows[qid], order[qid][:200]
        assert len(top200) == 200 and row["source"] == sources[qid]
        if not row["parsed"] and EXAMPLE_RE.search(queries[qid]):
            historical_pattern_only.append(qid)
        same_members += set(top200) == set(row["top200"])
        if top200 != row["top200"]:
            exact_ties = all(score_row[positions[a]] == score_row[positions[b]]
                             for a, b in zip(top200, row["top200"], strict=True))
            order_changes.append({"query_id": qid, "exact_ties_only": exact_ties, "production_top200": top200})
        for k, tie, score in zip(audit.CUTOFFS, row["cutoff_ties"], row["cutoff_scores"], strict=True):
            cutoff_ties[k] += tie
            tied_gold_excluded[k] += row["rank"] > k and row["gold_score"] == score
            rank = production[qid]["rank"]
            assert (rank is not None and rank <= k) == (row["rank"] <= k)

    def strata(field, test):
        return {name: audit.metrics(row for row in rows.values() if test(row[field]) == flag)
                for flag, name in ((True, "yes"), (False, "no"))}

    return {"measurement_sha256": audit.sha_file(measurements), "byte_identical_repeat": True,
            "production_mteb_recall": audit.metrics(production.values()),
            "production_gold_ranks": {qid: row["rank"] for qid, row in production.items()},
            "production_top200_order_matches": len(dev) - len(order_changes),
            "production_top200_membership_matches": same_members,
            "production_top200_exact_tie_only_changes": sum(row["exact_ties_only"] for row in order_changes),
            "production_top200_changes": order_changes,
            "cutoff_ties": dict(cutoff_ties), "shared_rank_extra_gold_hits": dict(tied_gold_excluded),
            "historical_regex_extra_query_ids": historical_pattern_only,
            "query_sample_overlap": len(set(sample) & set(dev)),
            "production_document_cache_query_hits": sum(
                text_key(MODEL_NAME, QUERY_PREFIX + queries[qid]) in cache._index for qid in dev),
            "recall_by_parsed": strata("parsed", bool),
            "recall_by_truncated": strata("formatted_tokens", lambda n: n > MAX_SEQ_LEN)}


def check_archive(cold_sample):
    history_dir = audit.HERE / "retrieval_consistency_run"
    first = history_dir / "historical_check.json"
    second = history_dir / "historical_repeat.json"
    assert audit.sha_file(first) == audit.sha_file(second)
    history = audit.read_json(first)
    assert history["production_full_gold_rank_matches_archive"] == history["production_full_top50_matches_archive"] == 500
    queries, _, _, sample, _, _, _ = audit.inputs()
    _, original = audit.read_vectors(history_dir / "sample_vectors.npz", sample, queries)
    _, cold = audit.read_vectors(cold_sample, sample, queries)
    return {"byte_identical_archived_replay": True, "archived_replay_sha256": audit.sha_file(first),
            "cold_sample_vector_sha256": cold["vector_sha256"],
            "cold_sample_matches_archived_vectors": cold["vector_sha256"] == original["vector_sha256"],
            "byte_identical_cold_sample_npz": audit.sha_file(cold_sample) == audit.sha_file(history_dir / "sample_vectors.npz"),
            "original_hits": history["metrics"]["production_full"]["recall_at_50"],
            "training_hits": history["metrics"]["fixed_remove_only"]["recall_at_50"],
            "gained_ids": history["entered_top50_query_ids"], "lost_ids": history["left_top50_query_ids"]}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("measurements", type=Path)
    cli.add_argument("repeat", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--vectors", type=Path, required=True)
    cli.add_argument("--cold-sample", type=Path, required=True)
    args = cli.parse_args()
    audit.configure()
    before = {file.name: audit.sha_file(file) for file in audit.FROZEN}
    report = {"dev_stdin": check_dev(args.measurements, args.repeat, args.vectors),
              "historical": check_archive(args.cold_sample)}
    assert before[EMB_CACHE.name] == audit.read_json(args.measurements)["corpus"]["cache_sha256"]
    assert before == {file.name: audit.sha_file(file) for file in audit.FROZEN}
    report["frozen_input_sha256"] = before
    audit.write_json(args.output, report)
