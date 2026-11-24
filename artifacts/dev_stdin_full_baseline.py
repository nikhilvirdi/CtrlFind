"""Save the source-stratified dev-stdin split and replay its unmodified full-corpus baseline."""
from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import random

import retrieval_consistency as audit
from src.cache import text_key
from src.config import MODEL_NAME, QUERY_PREFIX


VECTORS = audit.HERE / "retrieval_consistency_run" / "dev_vectors.npz"
NOTEBOOK = audit.ROOT / "notebook" / "baseline_recall_and_execution_validation.ipynb"


def historical_proof():
    notebook = audit.read_json(NOTEBOOK)
    evidence = {}
    for idx, token, line_starts in (
        (5, "corpus_t  = corpus[corpus.partition", ("corpus test :", "queries test:")),
        (9, "q_emb = model.encode(q_texts", ("shapes:",)),
        (11, "sim = q_emb @ c_emb.T", ("Recall@50 ",)),
    ):
        cell = notebook["cells"][idx]
        source = "".join(cell["source"])
        assert cell["cell_type"] == "code" and token in source
        # Retain only saved aggregate stdout, excluding any qrel display in the notebook.
        lines = [line.strip() for item in cell.get("outputs", []) if item.get("name") == "stdout"
                 for line in "".join(item.get("text", [])).splitlines() if line.strip().startswith(line_starts)]
        evidence[str(idx)] = {"source": source, "saved_aggregates": lines}
    assert evidence["5"]["saved_aggregates"] == ["corpus test : 3765", "queries test: 3765"]
    assert evidence["9"]["saved_aggregates"] == ["shapes: (3765, 768) (3765, 768)"]
    assert len(evidence["11"]["saved_aggregates"]) == 1 and "31.29%" in evidence["11"]["saved_aggregates"][0]
    return {"notebook": str(NOTEBOOK.relative_to(audit.ROOT).as_posix()), "notebook_sha256": audit.sha_file(NOTEBOOK),
            "candidate_count": 3765, "query_count": 3765, "recall_at_50_percent": 31.29, "cells_zero_based": evidence}


def stratified_split(dev, sources):
    halves = {"A": [], "B": []}
    odd = 0
    for source in sorted(audit.DEV_SOURCES):
        members = sorted(qid for qid in dev if sources[qid] == source)
        random.Random(f"{audit.SEED}:{source}").shuffle(members)
        extra_to_a = len(members) % 2 and odd % 2 == 0
        boundary = len(members) // 2 + extra_to_a
        odd += len(members) % 2
        halves["A"].extend(members[:boundary])
        halves["B"].extend(members[boundary:])
    a, b = halves.values()
    assert len(a) == len(b) == 873 and not set(a) & set(b) and set(a + b) == set(dev)
    return {"dataset_revision": audit.REVISION, "source_panel_sha256": audit.ids_digest(dev),
            "seed": audit.SEED, "assignment": "Sort IDs within source, shuffle with Random(f'{seed}:{source}'); "
            "alternate the extra odd-stratum member A then B in hostname order.",
            "halves": {arm: {"query_ids": ids, "query_ids_sha256": audit.ids_digest(ids),
                             "source_counts": dict(sorted(Counter(sources[qid] for qid in ids).items())),
                             "queries": len(ids)} for arm, ids in halves.items()}}


def calculate_metrics(rows):
    rows = list(rows)
    dcg = math.fsum(1.0 / math.log2(row["rank"] + 1)
                    for row in rows if row["rank"] is not None and row["rank"] <= 10)
    return {**audit.metrics(rows), "ndcg_at_10": {"percent": 100.0 * dcg / len(rows), "total": len(rows)}}


def baseline(data, vectors_path, split_path):
    from mteb.similarity_functions import cos_sim

    frozen = {path.name: audit.sha_file(path) for path in audit.FROZEN}
    queries, sources, gold, _, dev, old, ablation = data
    split = audit.read_json(split_path)
    assert split == stratified_split(dev, sources)
    vectors, manifest = audit.read_vectors(vectors_path, dev, queries)
    ids, train, _, passage_texts, _, passages, _, cache = audit.corpus(gold, old, ablation)
    assert len(dev) == 1746 and len(ids) == 8765 and len(train) == 5000
    assert {gold[qid] for qid in dev} <= set(train)

    scores = cos_sim(vectors, passages).numpy()
    rows = audit.fixed_ranks(scores, ids, dev, gold)
    assert rows == audit.fixed_ranks(scores[:, ::-1], ids[::-1], dev, gold)
    for qid, row in rows.items():
        row["source"] = sources[qid]

    production = audit.production_orders(dev, [queries[qid] for qid in dev], vectors, ids,
                                         passage_texts, passages, old["model"]["search_depth"])
    production_rows = {qid: {"rank": order.index(gold[qid]) + 1 if gold[qid] in order else None}
                       for qid, order in production.items()}
    production_cutoff_hit_differences = {
        k: [qid for qid in dev if (rows[qid]["rank"] <= k) !=
            (production_rows[qid]["rank"] is not None and production_rows[qid]["rank"] <= k)]
        for k in audit.CUTOFFS
    }
    assert frozen == {path.name: audit.sha_file(path) for path in audit.FROZEN}
    return {
        "historical": historical_proof(),
        "config": {"query_split": "train", "dataset_revision": audit.REVISION, "query_ids": dev,
                   "query_ids_sha256": audit.ids_digest(dev), "training_qrels_sha256": audit.sha_file(
                       audit.parquet("data/train-*.parquet")),
                   "split_file_sha256": audit.sha_file(split_path), "query_vector_manifest": manifest,
                   "candidate_corpus": {"document_ids_in_parquet_order": ids, "training_count": len(train),
                                        "total": len(ids), "ordered_id_sha256": audit.ids_digest(ids),
                                        "parquet_sha256": old["corpus"]["parquet_sha256"],
                                        "vector_sha256": audit.vector_digest(passages)},
                   "metrics": "One positive training qrel per query. Macro Recall at fixed K; binary NDCG@10.",
                   "score_order": "MTEB float32 cosine descending, lexical document ID ascending on exact ties.",
                   "search_depth": len(ids), "saved_candidate_depth": max(audit.CUTOFFS),
                   "candidate_cache_hits": len(ids), "production_cache_query_hits": sum(
                       text_key(MODEL_NAME, QUERY_PREFIX + queries[qid]) in cache for qid in dev),
                   "frozen_input_sha256_before_and_after": frozen},
        "metrics": calculate_metrics(rows.values()),
        "source_metrics": {source: calculate_metrics(r for r in rows.values() if r["source"] == source)
                           for source in sorted(audit.DEV_SOURCES)},
        "checks": {"reversed_corpus_same_rankings": True, "production_stage1": calculate_metrics(production_rows.values()),
                   "production_hit_differences": production_cutoff_hit_differences,
                   "production_top200_membership_matches": sum(set(production[qid][:200]) == set(rows[qid]["top200"])
                                                              for qid in dev),
                   "production_top200_order_matches": sum(production[qid][:200] == rows[qid]["top200"] for qid in dev),
                   "exact_cutoff_ties": dict(zip(audit.CUTOFFS, map(sum, zip(*(row["cutoff_ties"] for row in rows.values()))))),
                   "ordered_top200_sha256": audit.json_digest([(qid, rows[qid]["top200"]) for qid in dev])},
        "rows": rows,
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("split", "baseline"))
    cli.add_argument("output", type=Path)
    cli.add_argument("--vectors", type=Path, default=VECTORS)
    cli.add_argument("--split", type=Path)
    cli.add_argument("--reference", type=Path)
    args = cli.parse_args()
    audit.configure()
    data = audit.inputs()
    if args.mode == "split":
        _, sources, _, _, dev, *_ = data
        result = stratified_split(dev, sources)
    else:
        if args.split is None:
            cli.error("baseline requires --split")
        result = baseline(data, args.vectors, args.split)
        print(result["metrics"], flush=True)
    audit.write_json(args.output, result)
    if args.reference:
        assert audit.sha_file(args.output) == audit.sha_file(args.reference), "Repeat differs from reference"
        print("Byte-identical repeat", audit.sha_file(args.output), flush=True)
