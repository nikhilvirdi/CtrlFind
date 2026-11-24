"""Validate the historical training replay and score dev-stdin with train-only reads."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

import retrieval_consistency as audit
from src.cache import EmbeddingCache, text_key
from src.config import DOC_PREFIX, EMB_CACHE, MAX_SEQ_LEN, MODEL_NAME, QUERY_PREFIX
from src.execute import parse_example

VECTORS = audit.HERE / "retrieval_consistency_run"
HISTORY = VECTORS / "historical_check.json"


def training_documents(old, ablation):
    from mteb._create_dataloaders import _corpus_to_dict

    path = audit.parquet("corpus/*.parquet")
    rows = audit.pq.read_table(path, columns=["_id", "text", "title"],
                               filters=[("partition", "=", "train")]).to_pylist()
    rows = sorted(rows, key=lambda row: row["_id"])
    ids = [row["_id"] for row in rows]
    assert len(ids) == len(set(ids)) == ablation["corpus_count"] == 5000
    assert audit.ids_digest(ids) == ablation["corpus_sha256"]
    raw = [row["text"] for row in rows]
    formatted = [_corpus_to_dict({"id": row["_id"], **row})["text"] for row in rows]
    assert audit.sha_file(EMB_CACHE) == old["model"]["embedding_cache_sha256"]
    cache = EmbeddingCache(EMB_CACHE, MODEL_NAME)

    def cache_miss(texts):
        raise ValueError(f"{len(texts)} missing training document vectors")

    raw_vec, _ = cache.get_many([DOC_PREFIX + text for text in raw], cache_miss)
    formatted_vec, _ = cache.get_many([DOC_PREFIX + text for text in formatted], cache_miss)
    for vec in (raw_vec, formatted_vec):
        audit.validate_vectors(vec, len(ids))
    assert not cache.dirty
    return ids, raw, formatted, raw_vec, formatted_vec, cache


def boundary_delta(a, b, ids, gold_id):
    """Locate all changes above the gold or across rank 50."""
    order_a = np.argsort(-a, kind="stable")
    order_b = np.argsort(-b, kind="stable")
    gold_col = ids.index(gold_id)
    rank_a = int(np.flatnonzero(order_a == gold_col)[0]) + 1
    rank_b = int(np.flatnonzero(order_b == gold_col)[0]) + 1
    before_gold_a, before_gold_b = set(order_a[:rank_a - 1]), set(order_b[:rank_b - 1])
    top_a, top_b = set(order_a[:50]), set(order_b[:50])
    moved = {i for x, y in zip(order_a[:50], order_b[:50], strict=True) if x != y for i in (x, y)}
    relevant = (before_gold_a ^ before_gold_b) | moved | {gold_col}
    return {
        "gold_ranks": [rank_a, rank_b],
        "top50_a": [ids[i] for i in order_a[:50]], "top50_b": [ids[i] for i in order_b[:50]],
        "removed_above_gold": sorted(ids[i] for i in before_gold_a - before_gold_b),
        "added_above_gold": sorted(ids[i] for i in before_gold_b - before_gold_a),
        "top50_removed": sorted(ids[i] for i in top_a - top_b),
        "top50_added": sorted(ids[i] for i in top_b - top_a),
        "changed_top50_positions": {ids[i]: [int(np.flatnonzero(order == i)[0]) + 1 for order in (order_a, order_b)]
                                    for i in sorted(moved)},
        "changed_candidate_scores": {ids[i]: [float(a[i]), float(b[i])] for i in sorted(relevant)},
        "cutoff50_scores": [float(a[order_a[49]]), float(b[order_b[49]])],
    }


def attribute(output, vectors, history_path):
    from mteb.similarity_functions import cos_sim

    queries, _, gold, sample, _, old, ablation = audit.inputs()
    history = audit.read_json(history_path)
    assert history["query_ids"] == sample
    ids, raw, formatted, raw_vec, formatted_vec, cache = training_documents(old, ablation)
    train_ids = set(ids)
    assert train_ids == set(gold.values())
    qvec, manifest = audit.read_vectors(vectors / "sample_vectors.npz", sample, queries)
    query_hits = sum(text_key(MODEL_NAME, QUERY_PREFIX + queries[qid]) in cache._index for qid in sample)
    assert query_hits == history["historical_production_query_cache_hits"] == 0

    scores = cos_sim(qvec, formatted_vec).numpy()
    raw_cos = cos_sim(qvec, raw_vec).numpy()
    raw_dot = np.concatenate([qvec[i:i + audit.BATCH] @ raw_vec.T for i in range(0, len(sample), audit.BATCH)])
    stable = audit.fixed_ranks(scores, ids, sample, gold)
    assert stable == audit.fixed_ranks(scores[:, ::-1], ids[::-1], sample, gold)
    stable_raw_cos = audit.fixed_ranks(raw_cos, ids, sample, gold)
    stable_raw_dot = audit.fixed_ranks(raw_dot, ids, sample, gold)
    assert sum(a != b for a, b in zip(raw, formatted, strict=True)) == history["train_document_text_variants"]
    assert int(np.count_nonzero(np.any(raw_vec != formatted_vec, axis=1))) == history["train_document_vector_variants"]

    details = {}
    for qid in sample:
        witness = history["rows"][qid]
        old_row, train_row = old["rows"][qid], ablation["rows"][qid]
        before, after = witness["controlled_full"], witness["controlled_train"]
        assert history["query_text_sha256"][qid] == audit.sha_bytes(queries[qid].encode("utf-8"))
        assert witness["gold_id"] == gold[qid] == old_row["gold_id"]
        assert witness["production_full_rank"] == old_row["stage1_gold_rank"]
        assert witness["production_full_top50_matches_archive"] is True
        assert (gold[qid] in old_row["stage1_top50"]) == (before["rank"] <= 50)
        assert old_row["stage1_gold_rank"] == (before["rank"] if before["rank"] <= 1000 else None)
        assert witness["ablation_replay_rank"] == stable_raw_dot[qid]["rank"]
        assert witness["ablation_archived_rank"] == train_row["rank"]
        assert witness["ablation_top50_matches_archive"] == (stable_raw_dot[qid]["top200"][:50] == train_row["top"])
        assert before["rank"] - after["rank"] == witness["removed_distractors_ahead_of_gold"]
        assert after["rank"] == stable[qid]["rank"] == stable_raw_cos[qid]["rank"] == train_row["rank"]
        if before["rank"] <= 200:
            ahead = before["top200"][:before["rank"] - 1]
            assert sum(d not in train_ids for d in ahead) == witness["removed_distractors_ahead_of_gold"]
        details[qid] = {
            "gold_id": gold[qid], "source": witness["source"], "stable_full_rank": before["rank"],
            "archived_full_rank_at_depth_1000": old_row["stage1_gold_rank"],
            "distractors_removed_ahead_of_gold": witness["removed_distractors_ahead_of_gold"],
            "same_scores_training_rank": after["rank"], "training_cosine_rank": stable[qid]["rank"],
            "raw_document_cosine_rank": stable_raw_cos[qid]["rank"],
            "raw_document_dot_rank_shared_batches": stable_raw_dot[qid]["rank"],
            "original_training_rank": train_row["rank"],
            "query_text_sha256": history["query_text_sha256"][qid],
            "prefixed_query_sha256": audit.sha_bytes((QUERY_PREFIX + queries[qid]).encode("utf-8")),
        }
    exceptions = [qid for qid, row in details.items() if
                  len({row[key] for key in ("same_scores_training_rank", "training_cosine_rank",
                                            "raw_document_cosine_rank", "raw_document_dot_rank_shared_batches",
                                            "original_training_rank")}) > 1 or not history["rows"][qid]["ablation_top50_matches_archive"]]
    assert exceptions == ["q4336", "q4630"], exceptions
    e5 = audit.load_model(output.with_suffix(".unused.npz"))
    probes = {}
    for qid in exceptions:
        i = sample.index(qid)
        batch = next(row["ids"] for row in ablation["batches"] if qid in row["ids"])
        batch_i = batch.index(qid)
        original_qvec, misses = e5.encode_queries([queries[q] for q in batch])
        assert len(batch) == misses == audit.BATCH
        original_scores = (original_qvec @ raw_vec.T)[batch_i]
        batch_probe = boundary_delta(raw_dot[i], original_scores, ids, gold[qid])
        assert batch_probe["gold_ranks"][1] == ablation["rows"][qid]["rank"]
        assert batch_probe["top50_b"] == ablation["rows"][qid]["top"]
        batch_probe.update(batch_query_ids=batch, sample_vector_sha256=audit.vector_digest(qvec[i]),
                           historical_batch_vector_sha256=audit.vector_digest(original_qvec[batch_i]),
                           max_abs_query_vector_difference=float(np.max(np.abs(qvec[i] - original_qvec[batch_i]))))
        probes[qid] = {"encoding_batch": batch_probe, "cosine_vs_dot": boundary_delta(raw_cos[i], raw_dot[i], ids, gold[qid]),
                       "document_format": boundary_delta(scores[i], raw_cos[i], ids, gold[qid])}
    gained = [qid for qid, row in details.items() if row["same_scores_training_rank"] <= 50 < row["stable_full_rank"]]
    assert gained == history["entered_top50_query_ids"] and len(gained) == 34
    assert not history["left_top50_query_ids"]
    assert sum(row["stable_full_rank"] <= 50 for row in details.values()) == 311
    assert sum(row["original_training_rank"] <= 50 for row in details.values()) == 345
    assert all((row["original_training_rank"] <= 50) == (row["training_cosine_rank"] <= 50) for row in details.values())
    audit.write_json(output, {
        "dataset_revision": audit.REVISION, "sample_query_ids": sample, "sample_query_ids_sha256": audit.ids_digest(sample),
        "historical_replay_artifact": {"file": history_path.name, "sha256": audit.sha_file(history_path)},
        "training_sample_vector_manifest": manifest, "historical_metrics": history["metrics"],
        "corpus_removed_ahead_for_queries": sum(row["distractors_removed_ahead_of_gold"] > 0 for row in details.values()),
        "entered_top50_query_ids": gained, "entered_top50_by_source": history["entered_top50_by_source"],
        "prior_full_rank_and_top50_exact_matches": [history["production_full_gold_rank_matches_archive"],
                                                     history["production_full_top50_matches_archive"]],
        "training_rank_and_top50_matches_shared_batch": [history["shared_query_vectors_ablation_gold_rank_matches_archive"],
                                                          history["shared_query_vectors_ablation_top50_matches_archive"]],
        "training_rank_and_top50_matches_after_recreating_original_batches": [len(sample), len(sample)],
        "query_cache_hits": query_hits, "document_cache_hits_per_format": len(ids),
        "document_text_variants": history["train_document_text_variants"],
        "document_vector_variants": history["train_document_vector_variants"],
        "maximum_doc_vector_delta": history["max_abs_document_vector_variant_delta"],
        "cutoff_ties": history["cutoff_ties"], "cache_reencode_probe_from_historical_replay": history["document_cache_probe"],
        "numerical_controls": probes, "rows": details,
    })
    print(f"Reconciled 311 → 345 with {len(gained)} corpus-only crossings.")
    print(f"Original batches reproduce both archive exceptions: {exceptions}")


def score_dev(output, vectors, reference):
    from mteb.similarity_functions import cos_sim
    from transformers import AutoTokenizer

    queries, sources, gold, _, dev, old, ablation = audit.inputs()
    assert len(dev) == 1746
    ids, _, formatted, _, formatted_vec, _ = training_documents(old, ablation)
    assert set(ids) == set(gold.values())
    qvec, manifest = audit.read_vectors(vectors / "dev_vectors.npz", dev, queries)
    scores = cos_sim(qvec, formatted_vec).numpy()
    rows = audit.fixed_ranks(scores, ids, dev, gold)
    assert rows == audit.fixed_ranks(scores[:, ::-1], ids[::-1], dev, gold)
    tokenizer = AutoTokenizer.from_pretrained(audit.MODEL, local_files_only=True)
    assert tokenizer.truncation_side == "right"
    for start in range(0, len(dev), 64):
        batch = dev[start:start + 64]
        raw = [queries[qid] for qid in batch]
        full = [QUERY_PREFIX + text for text in raw]
        raw_tokens = tokenizer(raw, add_special_tokens=False, truncation=False)["input_ids"]
        full_tokens = tokenizer(full, add_special_tokens=True, truncation=False)["input_ids"]
        effective = tokenizer(full, add_special_tokens=True, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]
        for qid, text, tokens, prefixed, kept in zip(batch, raw, raw_tokens, full_tokens, effective, strict=True):
            assert len(kept) == min(len(prefixed), MAX_SEQ_LEN)
            rows[qid].update(source=sources[qid], characters=len(text), parsed=parse_example(text) is not None,
                             raw_tokens=len(tokens), formatted_tokens=len(prefixed), effective_tokens=len(kept),
                             raw_text_sha256=audit.sha_bytes(text.encode("utf-8")),
                             prefixed_text_sha256=audit.sha_bytes((QUERY_PREFIX + text).encode("utf-8")))
    result = {
        "dataset": audit.DATASET, "dataset_revision": audit.REVISION, "split": "train",
        "subset": "dev-stdin", "included_sources": sorted(audit.DEV_SOURCES),
        "excluded_sources": sorted(audit.EXCLUDED_SOURCES),
        "source_share": {source: audit.rate(sum(sources[qid] == source for qid in dev), len(dev))
                         for source in sorted(audit.DEV_SOURCES)},
        "all_training_source_counts": dict(sorted(Counter(sources.values()).items())),
        "membership": {"query_ids": dev, "ordered_ids_sha256": audit.ids_digest(dev)},
        "corpus": {"document_ids": ids, "ordered_ids_sha256": audit.ids_digest(ids),
                   "ordered_text_sha256": audit.json_digest(list(zip(ids, formatted, strict=True))),
                   "document_prefix": DOC_PREFIX, "vector_sha256": audit.vector_digest(formatted_vec),
                   "cache_sha256": old["model"]["embedding_cache_sha256"], "cache_hits": len(ids)},
        "qrels_sha256": audit.sha_file(audit.parquet("data/train-*.parquet")),
        "query_vector_manifest": manifest, "cutoffs": audit.CUTOFFS,
        "ranking": "All 5000 train scores, MTEB float32 cosine descending, lexical document ID ascending on exact ties",
        "metric": "Macro mean of 1[positive gold rank <= K] over all queries in each group",
        "summary": audit.subset_summary(rows.values()),
        "by_source": {source: audit.subset_summary(row for row in rows.values() if row["source"] == source)
                      for source in sorted(audit.DEV_SOURCES)},
        "top200_sha256": audit.json_digest([(qid, rows[qid]["top200"]) for qid in dev]),
        "rows": rows,
    }
    audit.write_json(output, result)
    if reference is not None:
        assert audit.sha_file(reference) == audit.sha_file(output), "Replay differs from reference"
        print(f"Verified byte-identical repeat: {audit.sha_file(output)}")
    print(result["summary"])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("attribute", "dev"))
    cli.add_argument("output", type=Path)
    cli.add_argument("--vectors", type=Path, default=VECTORS)
    cli.add_argument("--history", type=Path, default=HISTORY)
    cli.add_argument("--reference", type=Path)
    args = cli.parse_args()
    audit.configure()
    if args.mode == "attribute":
        attribute(args.output, args.vectors, args.history)
    else:
        score_dev(args.output, args.vectors, args.reference)
