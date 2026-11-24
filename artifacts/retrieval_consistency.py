"""Offline stage-1 regression and dev-stdin census. Only training queries and qrels."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
from statistics import median
import sys
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlsplit

os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")

import numpy as np
import pyarrow.parquet as pq
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src.cache import EmbeddingCache, text_key
from src.config import DATASET, DOC_PREFIX, EMB_CACHE, MAX_SEQ_LEN, MODEL_NAME, QUERY_PREFIX
from src.encoder import E5
from src.execute import parse_example

REVISION = "f22508f96b7a36c2415181ed8bb76f76e04ae2d5"
MODEL_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
HUB = Path.home() / ".cache/huggingface/hub"
DATA = HUB / "datasets--CoIR-Retrieval--apps/snapshots" / REVISION
MODEL = HUB / "models--intfloat--e5-base-v2/snapshots" / MODEL_REVISION
DEV_SOURCES = {"atcoder.jp", "codechef.com", "codeforces.com", "hackerrank.com"}
EXCLUDED_SOURCES = {"codewars.com", "leetcode.com"}
SEED, BATCH, THREADS = 20261125, 16, 4
CUTOFFS = (10, 50, 100, 200)
ALL_LEDGER = HERE / "retrieval_diagnostic_train.json"
TRAIN_LEDGER = HERE / "query_example_baseline_train.json"
FROZEN = (EMB_CACHE, HERE / "baseline_frozen.json", HERE / "appsretrieval_results.json",
          HERE / "exec_results.jsonl", HERE / "doc_ids.npy", ALL_LEDGER, TRAIN_LEDGER)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_digest(value):
    return sha_bytes(json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("ascii"))


def ids_digest(ids):
    return sha_bytes("\n".join(ids).encode("utf-8"))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(value, out, ensure_ascii=True, separators=(",", ":"))
        out.write("\n")


def parquet(pattern):
    (path,) = DATA.glob(pattern)
    return path


def inputs():
    train = [("partition", "=", "train")]
    query_file = parquet("queries/*.parquet")
    qrel_file = parquet("data/train-*.parquet")
    rows = pq.read_table(query_file, columns=["_id", "text", "meta_information"], filters=train).to_pylist()
    labels = pq.read_table(qrel_file, columns=["query-id", "corpus-id", "score"]).to_pylist()
    queries = {row["_id"]: row["text"] for row in rows}
    sources = {row["_id"]: urlsplit(row["meta_information"]["url"]).hostname.removeprefix("www.")
               for row in rows}
    gold = {row["query-id"]: row["corpus-id"] for row in labels if row["score"] > 0}
    assert len(rows) == len(queries) == len(labels) == len(gold) == len(set(gold.values())) == 5000
    assert gold.keys() == queries.keys() and set(sources.values()) == DEV_SOURCES | EXCLUDED_SOURCES
    old, ablation = read_json(ALL_LEDGER), read_json(TRAIN_LEDGER)
    assert old["query_split"] == "train" and old["dataset_revision"] == ablation["train_revision"] == REVISION
    assert old["model"]["revision"] == ablation["model_revision"] == MODEL_REVISION
    assert (old["model"]["query_prefix"], old["model"]["document_prefix"], old["model"]["max_seq_length"]) == (
        QUERY_PREFIX, DOC_PREFIX, MAX_SEQ_LEN)
    assert old["model"]["normalize_embeddings"] and old["model"]["batch_size"] == ablation["batch_size"] == BATCH
    sample = random.Random(SEED).sample(sorted(queries), 500)
    assert sample == old["sampling"]["query_ids"] == ablation["populations"]["uniform"]
    assert all(old["rows"][qid]["gold_id"] == gold[qid] for qid in sample)
    dev = [qid for qid in sorted(queries) if sources[qid] in DEV_SOURCES]
    return queries, sources, gold, sample, dev, old, ablation


def package_versions():
    return {name: version(name) for name in ("datasets", "mteb", "numpy", "pyarrow",
            "sentence-transformers", "tokenizers", "torch", "transformers")}


def model_files():
    return {p.relative_to(MODEL).as_posix(): sha_file(p) for p in sorted(MODEL.rglob("*"))
            if p.is_file() and p.suffix != ".md"}


def configure():
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def load_model(cache_path):
    from sentence_transformers import SentenceTransformer

    e5 = E5.__new__(E5)
    e5.device, e5.batch_size = "cpu", BATCH
    e5.model = SentenceTransformer(str(MODEL), device="cpu", local_files_only=True)
    e5.model.max_seq_length = MAX_SEQ_LEN
    e5.cache = EmbeddingCache(cache_path, MODEL_NAME)
    assert len(e5.cache) == 0 and e5.model.default_prompt_name is None
    return e5


def vector_digest(vectors):
    return sha_bytes(np.ascontiguousarray(vectors, dtype="<f4").tobytes())


def validate_vectors(vectors, count):
    assert vectors.shape == (count, 768) and vectors.dtype == np.float32
    assert np.isfinite(vectors).all()
    assert np.max(np.abs(np.linalg.norm(vectors, axis=1) - 1)) < 2e-6


def cache_identity(qids, queries):
    return {"model": MODEL_NAME, "revision": MODEL_REVISION, "files": model_files(),
            "prefix": QUERY_PREFIX, "max_tokens": MAX_SEQ_LEN, "batch_size": BATCH, "threads": THREADS,
            "device": "cpu", "dtype": "float32", "normalize_embeddings": True,
            "packages": package_versions(), "encoder_sha256": sha_file(ROOT / "src/encoder.py"),
            "ordered_ids_sha256": ids_digest(qids), "ordered_inputs_sha256": json_digest(
                [(qid, QUERY_PREFIX + queries[qid]) for qid in qids])}


def write_vectors(path, qids, queries, vectors):
    validate_vectors(vectors, len(qids))
    manifest = {**cache_identity(qids, queries), "vector_sha256": vector_digest(vectors)}
    with path.open("xb") as out:
        np.savez_compressed(out, ids=qids, vectors=vectors, manifest=json.dumps(manifest, sort_keys=True))
    print(f"Wrote {path.name}: {manifest['vector_sha256']}", flush=True)


def encode(output):
    queries, _, _, sample, dev, *_ = inputs()
    sample_path = output / "sample_vectors.npz"
    dev_path = output / "dev_vectors.npz"
    if sample_path.exists() or dev_path.exists():
        raise FileExistsError("Use an empty output directory for cold encoding")
    e5 = load_model(output / "unused.npz")
    sample_vectors, _ = e5.encode_queries([queries[qid] for qid in sample])
    write_vectors(sample_path, sample, queries, sample_vectors)
    # Reuse exact content hits from the sample. E5 sorts all remaining texts for batching.
    dev_vectors, misses = e5.encode_queries([queries[qid] for qid in dev])
    keys = {text_key(MODEL_NAME, QUERY_PREFIX + queries[qid]) for qid in sample}
    assert misses == len({text_key(MODEL_NAME, QUERY_PREFIX + queries[qid]) for qid in dev} - keys)
    write_vectors(dev_path, dev, queries, dev_vectors)


def read_vectors(path, qids, queries):
    with np.load(path, allow_pickle=False) as saved:
        assert saved["ids"].tolist() == qids
        vectors = saved["vectors"]
        manifest = json.loads(saved["manifest"].item())
    validate_vectors(vectors, len(qids))
    assert manifest == {**cache_identity(qids, queries), "vector_sha256": vector_digest(vectors)}
    return vectors, manifest


def corpus(gold, old, ablation):
    from mteb._create_dataloaders import _corpus_to_dict

    path = parquet("corpus/*.parquet")
    physical_rows = pq.read_table(path, columns=["_id", "text", "title", "partition"]).to_pylist()
    rows = {row["_id"]: row for row in physical_rows}
    full_ids = list(rows)
    train_ids = sorted(did for did, row in rows.items() if row["partition"] == "train")
    assert len(physical_rows) == len(rows) == old["corpus"]["count"] == 8765
    assert set(train_ids) == set(gold.values()) and len(train_ids) == ablation["corpus_count"]
    assert sha_file(path) == old["corpus"]["parquet_sha256"]
    assert ids_digest(full_ids) == old["corpus"]["ordered_doc_ids_sha256"]
    assert ids_digest(train_ids) == ablation["corpus_sha256"]
    raw_train = [rows[did]["text"] for did in train_ids]
    mteb_texts = [_corpus_to_dict({"id": did, **rows[did]})["text"] for did in full_ids]
    assert sha_file(EMB_CACHE) == old["model"]["embedding_cache_sha256"]
    with np.load(EMB_CACHE, allow_pickle=False) as saved:
        keys = saved["keys"].tolist()
        vectors = saved["vecs"]
    validate_vectors(vectors, len(keys))
    index = dict(zip(keys, range(len(keys)), strict=True))
    assert len(index) == len(keys)

    def cached(texts):
        return vectors[[index[text_key(MODEL_NAME, DOC_PREFIX + text)] for text in texts]]

    physical_index = {did: i for i, did in enumerate(full_ids)}
    train_cols = [physical_index[did] for did in train_ids]
    return full_ids, train_ids, train_cols, mteb_texts, raw_train, cached(mteb_texts), cached(raw_train), index


def fixed_ranks(scores, ids, qids, gold):
    positions = {did: n for n, did in enumerate(ids)}
    assert scores.shape == (len(qids), len(ids)) and np.isfinite(scores).all()
    assert len(positions) == len(ids) and len(set(qids)) == len(qids)
    lex = np.argsort(np.asarray(ids), kind="stable")
    score_order = lex[np.argsort(-scores[:, lex], axis=1, kind="stable")]
    rows = {}
    for qid, order, values in zip(qids, score_order, scores, strict=True):
        gold_col = positions[gold[qid]]
        rows[qid] = {
            "gold_id": gold[qid], "rank": int(np.flatnonzero(order == gold_col)[0]) + 1,
            "top200": [ids[i] for i in order[:200]], "gold_score": float(values[gold_col]),
            "cutoff_scores": [float(values[order[k - 1]]) for k in CUTOFFS],
            "cutoff_ties": [bool(values[order[k - 1]] == values[order[k]]) for k in CUTOFFS],
        }
    return rows


def production_orders(qids, query_texts, qvec, ids, doc_texts, doc_vec, depth):
    from datasets import Dataset
    from mteb import get_task
    from mteb.models.search_wrappers import SearchEncoderWrapper
    from mteb.types import PromptType
    from src.mteb_wrappers import E5Encoder

    def cached_encode(inputs, *, prompt_type, **_):
        texts = [text for batch in inputs for text in batch["text"]]
        expected, vectors = (query_texts, qvec) if prompt_type == PromptType.query else (doc_texts, doc_vec)
        assert texts == expected
        return vectors

    encoder = E5Encoder(SimpleNamespace(model=None), "prism/e5-base-v2-baseline")
    options = dict(task_metadata=get_task("AppsRetrieval").metadata, hf_split="train", hf_subset="default",
                   encode_kwargs={"batch_size": BATCH})
    search = SearchEncoderWrapper(encoder)
    search.index(Dataset.from_dict({"id": ids, "text": doc_texts}), **options)
    with patch.object(encoder, "encode", side_effect=cached_encode):
        hits = search.search(Dataset.from_dict({"id": qids, "text": query_texts}), top_k=depth, **options)
    return {qid: sorted(scores, key=lambda did: -scores[did]) for qid, scores in hits.items()}


def rate(count, n):
    return {"count": count, "total": n, "percent": 100 * count / n}


def metrics(rows):
    rows = list(rows)
    return {f"recall_at_{k}": rate(sum(row["rank"] is not None and row["rank"] <= k for row in rows), len(rows))
            for k in CUTOFFS}


def cutoff_ties(rows):
    return {str(k): sum(row["cutoff_ties"][i] for row in rows.values()) for i, k in enumerate(CUTOFFS)}


def compare_history(data, qvec, corpus_data):
    from mteb.similarity_functions import cos_sim

    queries, sources, gold, sample, _, old, ablation = data
    full_ids, train_ids, train_cols, mteb_texts, raw_train, full_vec, raw_vec, cache_index = corpus_data
    query_cache_hits = sum(text_key(MODEL_NAME, QUERY_PREFIX + queries[qid]) in cache_index for qid in sample)
    assert query_cache_hits == 0
    cosine = cos_sim(qvec, full_vec).numpy()
    production = production_orders(sample, [queries[qid] for qid in sample], qvec, full_ids,
                                   mteb_texts, full_vec, old["model"]["search_depth"])
    train_dot = np.concatenate([qvec[i:i+BATCH] @ raw_vec.T for i in range(0, len(sample), BATCH)])
    controlled = fixed_ranks(cosine[:, train_cols], train_ids, sample, gold)
    official_fixed = fixed_ranks(cosine, full_ids, sample, gold)
    train_legacy = fixed_ranks(train_dot, train_ids, sample, gold)
    recomputed = fixed_ranks(cos_sim(qvec, full_vec[train_cols]).numpy(), train_ids, sample, gold)
    raw_cos = fixed_ranks(cos_sim(qvec, raw_vec).numpy(), train_ids, sample, gold)
    # This control reverses file order while keeping every document score fixed.
    reversed_order = fixed_ranks(cosine[:, ::-1], full_ids[::-1], sample, gold)
    assert reversed_order == official_fixed
    train_set = set(train_ids)
    audits = {}
    for qid in sample:
        archived, train_archived = old["rows"][qid], ablation["rows"][qid]
        order, did = production[qid], gold[qid]
        rank = order.index(did) + 1 if did in order else None
        before, after = official_fixed[qid], controlled[qid]
        assert before["rank"] >= after["rank"]
        assert after["top200"][:sum(d in train_set for d in before["top200"])] == [
            d for d in before["top200"] if d in train_set]
        audits[qid] = {"source": sources[qid], "gold_id": did,
                       "production_full_rank": rank, "archived_full_rank": archived["stage1_gold_rank"],
                       "production_full_top50_matches_archive": order[:50] == archived["stage1_top50"],
                       "ablation_replay_rank": train_legacy[qid]["rank"], "ablation_archived_rank": train_archived["rank"],
                       "ablation_top50_matches_archive": train_legacy[qid]["top200"][:50] == train_archived["top"],
                       "controlled_full": before, "controlled_train": after,
                       "removed_distractors_ahead_of_gold": before["rank"] - after["rank"],
                       "removed_distractors_in_full_top50": [d for d in before["top200"][:50] if d not in train_set]}
    gained = [qid for qid in sample if controlled[qid]["rank"] <= 50 < official_fixed[qid]["rank"]]
    lost = [qid for qid in sample if official_fixed[qid]["rank"] <= 50 < controlled[qid]["rank"]]
    old_hits = {qid for qid in sample if gold[qid] in old["rows"][qid]["stage1_top50"]}
    ablation_hits = {qid for qid in sample if gold[qid] in ablation["rows"][qid]["top"]}
    assert not lost and len(gained) == 34 and set(gained) == ablation_hits - old_hits
    assert len(old_hits) == 311 and len(ablation_hits) == 345 and old_hits <= ablation_hits
    assert old_hits == {qid for qid in sample if gold[qid] in production[qid][:50]}
    assert old_hits == {qid for qid in sample if official_fixed[qid]["rank"] <= 50}
    assert all(ablation_hits == {qid for qid in sample if rows[qid]["rank"] <= 50}
               for rows in (controlled, train_legacy, recomputed, raw_cos))
    train_summary = {name: metrics(rows.values()) for name, rows in
                     (("fixed_full_cosine", official_fixed), ("fixed_remove_only", controlled),
                      ("mteb_format_rescored_train_cosine", recomputed), ("raw_train_cosine", raw_cos),
                      ("raw_train_dot", train_legacy))}
    train_summary["production_full"] = metrics({"rank": row["production_full_rank"]} for row in audits.values())
    return {
        "query_ids": sample, "query_ids_sha256": ids_digest(sample), "metrics": train_summary,
        "entered_top50_query_ids": gained, "left_top50_query_ids": lost,
        "entered_top50_by_source": dict(sorted(Counter(sources[qid] for qid in gained).items())),
        "production_full_gold_rank_matches_archive": sum(r["production_full_rank"] == r["archived_full_rank"]
                                                          for r in audits.values()),
        "shared_query_vectors_ablation_gold_rank_matches_archive": sum(
            r["ablation_replay_rank"] == r["ablation_archived_rank"] for r in audits.values()),
        "production_full_top50_matches_archive": sum(r["production_full_top50_matches_archive"] for r in audits.values()),
        "shared_query_vectors_ablation_top50_matches_archive": sum(
            r["ablation_top50_matches_archive"] for r in audits.values()),
        "cutoff_ties": {"full": cutoff_ties(official_fixed), "train": cutoff_ties(controlled)},
        "train_document_text_variants": sum(mteb_texts[i] != raw for i, raw in zip(train_cols, raw_train, strict=True)),
        "train_document_vector_variants": int(np.count_nonzero(np.any(full_vec[train_cols] != raw_vec, axis=1))),
        "max_abs_document_vector_variant_delta": float(np.max(np.abs(full_vec[train_cols] - raw_vec))),
        "historical_production_query_cache_hits": query_cache_hits,
        "rows": audits,
    }


def original_ablation_batches(history, data, sample_vec, corpus_data, e5):
    queries, _, gold, sample, _, _, ablation = data
    _, train_ids, _, _, _, _, raw_vec, _ = corpus_data
    sample_vectors = dict(zip(sample, sample_vec, strict=True))
    changed = {qid for qid, row in history["rows"].items() if not row["ablation_top50_matches_archive"]
               or row["ablation_replay_rank"] != row["ablation_archived_rank"]}
    cold = {}
    for batch in ablation["batches"]:
        ids = batch["ids"]
        affected = changed.intersection(ids)
        if not affected:
            continue
        e5.cache = EmbeddingCache(e5.cache.path, MODEL_NAME)
        assert len(e5.cache) == 0
        fresh, misses = e5.encode_queries([queries[qid] for qid in ids])
        assert misses == batch["misses"]
        original = fixed_ranks(fresh @ raw_vec.T, train_ids, ids, gold)
        shared = np.stack([sample_vectors.get(qid, vector) for qid, vector in zip(ids, fresh, strict=True)])
        shared_rows = fixed_ranks(shared @ raw_vec.T, train_ids, ids, gold)
        for qid in ids:
            if qid not in affected:
                continue
            row, recorded = original[qid], ablation["rows"][qid]
            assert (row["rank"], row["top200"][:50]) == (recorded["rank"], recorded["top"])
            assert shared_rows[qid]["rank"] == history["rows"][qid]["ablation_replay_rank"]
            cold[qid] = {"original_batch_query_ids": ids, "original_batch_vectors_sha256": vector_digest(fresh),
                         "rank_shared_encoding": shared_rows[qid]["rank"], "rank_original_encoding": row["rank"],
                         "top50_shared_encoding": shared_rows[qid]["top200"][:50],
                         "top50_original_encoding": row["top200"][:50],
                         "max_abs_query_vector_delta": float(np.max(np.abs(fresh[ids.index(qid)] - sample_vectors[qid])))}
    assert cold.keys() == changed
    return cold


def document_cache_probe(corpus_data, e5):
    _, train_ids, train_cols, full_text, _, full_vec, _, _ = corpus_data
    sample_cols = random.Random(SEED).sample(train_cols, BATCH)
    e5.cache = EmbeddingCache(e5.cache.path, MODEL_NAME)
    fresh, misses = e5.encode_docs([full_text[i] for i in sample_cols])
    assert misses == BATCH
    expected = full_vec[sample_cols]
    delta = np.abs(fresh - expected)
    proximity = np.sum(fresh * expected, axis=1) / (np.linalg.norm(fresh, axis=1) * np.linalg.norm(expected, axis=1))
    assert float(delta.max()) < 1e-5
    doc_ids = [train_ids[train_cols.index(i)] for i in sample_cols]
    return {"doc_ids": doc_ids, "fresh_vectors_sha256": vector_digest(fresh),
            "cached_vectors_sha256": vector_digest(expected), "max_absolute_delta": float(delta.max()),
            "min_cosine": float(proximity.min())}


def token_lengths(e5, qids, queries):
    results = {}
    for i in range(0, len(qids), 64):
        ids = qids[i:i+64]
        raw = [queries[qid] for qid in ids]
        prefixed = [QUERY_PREFIX + text for text in raw]
        raw_tokens = e5.model.tokenizer(raw, add_special_tokens=False, truncation=False)["input_ids"]
        full_tokens = e5.model.tokenizer(prefixed, add_special_tokens=True, truncation=False)["input_ids"]
        effective = e5.model[0].preprocess(prefixed)["attention_mask"].sum(dim=1).tolist()
        assert effective == [min(MAX_SEQ_LEN, len(tokens)) for tokens in full_tokens]
        results.update(zip(ids, zip(map(len, raw_tokens), map(len, full_tokens), effective, strict=True), strict=True))
    return results


def subset_summary(rows):
    rows = list(rows)
    return {"queries": len(rows), "recall": metrics(rows),
            "parser_coverage": rate(sum(r["parsed"] for r in rows), len(rows)),
            "over_512_formatted_tokens": rate(sum(r["formatted_tokens"] > MAX_SEQ_LEN for r in rows), len(rows)),
            "lengths": {key: {"median": median(r[key] for r in rows),
                              "p95": float(np.percentile([r[key] for r in rows], 95)),
                              "max": max(r[key] for r in rows)}
                        for key in ("characters", "raw_tokens", "formatted_tokens", "effective_tokens")}}


def evaluate_dev(data, qvec, corpus_data, e5):
    from mteb.similarity_functions import cos_sim

    queries, sources, gold, sample, dev, old, _ = data
    _, train_ids, train_cols, full_texts, _, full_vec, _, _ = corpus_data
    train_vec = full_vec[train_cols]
    scores = cos_sim(qvec, train_vec).numpy()
    ranks = fixed_ranks(scores, train_ids, dev, gold)
    production = production_orders(dev, [queries[qid] for qid in dev], qvec, train_ids,
                                   [full_texts[i] for i in train_cols], train_vec, old["model"]["search_depth"])
    lengths = token_lengths(e5, list(dict.fromkeys(sample + dev)), queries)
    assert all(list(lengths[qid]) == old["rows"][qid]["query_tokens"] for qid in sample)
    assert all((parse_example(queries[qid]) is not None) == old["rows"][qid]["example_found"] for qid in sample)
    prod_ranks = {}
    for qid in dev:
        raw, formatted, effective = lengths[qid]
        text, order = queries[qid], production[qid]
        production_rank = order.index(gold[qid]) + 1 if gold[qid] in order else None
        prod_ranks[qid] = {"rank": production_rank}
        ranks[qid].update(source=sources[qid], characters=len(text), raw_tokens=raw, formatted_tokens=formatted,
                          effective_tokens=effective, parsed=parse_example(text) is not None,
                          production_gold_rank=production_rank,
                          raw_text_sha256=sha_bytes(text.encode("utf-8")),
                          formatted_text_sha256=sha_bytes((QUERY_PREFIX + text).encode("utf-8")))
    summary = subset_summary(ranks.values())
    disagreements = {str(k): [qid for qid, row in ranks.items() if (row["rank"] <= k) != (
        row["production_gold_rank"] is not None and row["production_gold_rank"] <= k)] for k in CUTOFFS}
    return {"query_ids": dev, "query_ids_sha256": ids_digest(dev), "summary": summary,
            "sources": {source: subset_summary(row for row in ranks.values() if row["source"] == source)
                        for source in sorted(DEV_SOURCES)},
            "recall_by_truncation": {name: metrics(row for row in ranks.values()
                                                    if (row["formatted_tokens"] > MAX_SEQ_LEN) == truncated)
                                     for name, truncated in (("over_512", True), ("at_most_512", False))},
            "exact_cutoff_ties": cutoff_ties(ranks),
            "training_source_counts": dict(sorted(Counter(sources.values()).items())),
            "production_top200_order_matches": sum(production[qid][:200] == ranks[qid]["top200"] for qid in dev),
            "production_recall": metrics(prod_ranks.values()),
            "production_cutoff_gold_membership_disagreements": disagreements,
            "top200_sha256": json_digest([(qid, ranks[qid]["top200"]) for qid in dev]),
            "rows": ranks}


def replay(vectors, output, reference, historical_only):
    before = {path.name: sha_file(path) for path in FROZEN}
    data = inputs()
    queries, _, gold, sample, dev, old, ablation = data
    corpus_data = corpus(gold, old, ablation)
    qvec, sample_manifest = read_vectors(vectors / "sample_vectors.npz", sample, queries)
    history = compare_history(data, qvec, corpus_data)
    full_ids, train_ids, train_cols, mteb_texts, _, full_vec, _, _ = corpus_data
    history["query_text_sha256"] = {qid: sha_bytes(queries[qid].encode("utf-8")) for qid in sample}
    history["fixed_top200_sha256"] = {
        mode: json_digest([(qid, history["rows"][qid][mode]["top200"]) for qid in sample])
        for mode in ("controlled_full", "controlled_train")}
    print(json.dumps({key: val for key, val in history.items()
                      if key in ("metrics", "entered_top50_by_source", "cutoff_ties") or "matches_archive" in key}), flush=True)
    e5 = load_model(output.with_name("unused.npz"))
    history["original_ablation_batch_checks"] = original_ablation_batches(history, data, qvec, corpus_data, e5)
    cold_checked = history["original_ablation_batch_checks"].keys()
    history["ablation_corrected_gold_rank_matches_archive"] = sum(
        row["ablation_replay_rank"] == row["ablation_archived_rank"] or qid in cold_checked
        for qid, row in history["rows"].items())
    history["ablation_corrected_top50_matches_archive"] = sum(
        row["ablation_top50_matches_archive"] or qid in cold_checked for qid, row in history["rows"].items())
    history["document_cache_probe"] = document_cache_probe(corpus_data, e5)
    assert before == {path.name: sha_file(path) for path in FROZEN}
    if historical_only:
        write_json(output, history)
        return
    dev_vec, dev_manifest = read_vectors(vectors / "dev_vectors.npz", dev, queries)
    subset = evaluate_dev(data, dev_vec, corpus_data, e5)
    assert before == {path.name: sha_file(path) for path in FROZEN}
    result = {
        "configuration": {
            "dataset": DATASET, "revision": REVISION, "split": "train",
            "query_inputs_sha256": json_digest([(qid, queries[qid]) for qid in sorted(queries)]),
            "training_qrels_sha256": sha_file(parquet("data/train-*.parquet")),
            "candidate_parquet_sha256": old["corpus"]["parquet_sha256"], "sample_seed": SEED,
            "dev_sources": sorted(DEV_SOURCES), "excluded_sources": sorted(EXCLUDED_SOURCES),
            "full_document_ids_in_parquet_order": full_ids, "training_document_ids_in_lexical_order": train_ids,
            "full_document_ids_sha256": ids_digest(full_ids), "training_document_ids_sha256": ids_digest(train_ids),
            "full_document_source_sha256": json_digest(list(zip(full_ids, mteb_texts, strict=True))),
            "training_document_source_sha256": json_digest([(did, mteb_texts[i]) for did, i in zip(train_ids, train_cols, strict=True)]),
            "full_document_vector_sha256": vector_digest(full_vec),
            "training_document_vector_sha256": vector_digest(full_vec[train_cols]),
            "sample_vectors": sample_manifest, "dev_vectors": dev_manifest,
            "encoding_schedule": "Cold sample first as one E5 encode call. Sorted dev IDs next, with exact sample content hits reused.",
            "sample_dev_query_overlap": len(set(sample) & set(dev)),
            "cutoffs": CUTOFFS, "official_search_depth": old["model"]["search_depth"],
            "document_prefix": DOC_PREFIX, "query_prefix": QUERY_PREFIX, "interop_threads": 1,
            "ranking": "MTEB float32 cosine; descending score then ascending lexical doc ID; all scores eligible",
            "relevance": "Exactly one positive CoIR training qrel per query. All other IDs are nonrelevant.",
            "parser": "src.execute.parse_example on the original untruncated query; returned pair is a hit",
            "platform": platform.platform(), "python": sys.version,
            "production_source_sha256": {str(p.relative_to(ROOT).as_posix()): sha_file(p) for p in
                                        (ROOT / "src/config.py", ROOT / "src/cache.py", ROOT / "src/encoder.py",
                                         ROOT / "src/mteb_wrappers.py", ROOT / "src/execute.py")},
            "frozen_inputs_sha256_before_and_after": before,
        },
        "discrepancy": history, "dev_stdin": subset,
    }
    write_json(output, result)
    if reference is not None:
        assert sha_file(output) == sha_file(reference), "Full replay output differs from the reference"
        print(f"Byte-identical replay: {sha_file(output)}", flush=True)
    print(json.dumps(subset["summary"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["encode", "replay"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--vectors", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--historical-only", action="store_true")
    args = parser.parse_args()
    configure()
    if args.mode == "encode":
        args.output.mkdir(parents=True, exist_ok=True)
        encode(args.output)
    else:
        if args.vectors is None:
            parser.error("replay requires --vectors")
        if args.reference is not None and args.historical_only:
            parser.error("a reference comparison requires the complete replay")
        replay(args.vectors, args.output, args.reference, args.historical_only)
