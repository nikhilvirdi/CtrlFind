"""Measure MTEB E5 retrieval and the live top-50 verification funnel on train queries."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.parse import urlsplit

from verification_research import ROOT, SNAPSHOT, colon_candidates, load_train
from src import execute
from src.config import DATASET, DEFAULT_K, DOC_PREFIX, EMB_CACHE, EXEC_TIMEOUT, MAX_SEQ_LEN, MODEL_NAME, QUERY_PREFIX
from src.encoder import E5

MODEL_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
SEED = 20261125
SAMPLE_SIZE = 500
CUTOFFS = (10, 50, 100, 200)
THREADS = 4


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_lengths(model, raw, formatted):
    tokenizer = model.tokenizer
    rows = []
    for start in range(0, len(raw), 64):
        batch = formatted[start:start + 64]
        before = tokenizer(raw[start:start + 64], add_special_tokens=False, truncation=False)["input_ids"]
        full = tokenizer(batch, add_special_tokens=True, truncation=False)["input_ids"]
        kept = model[0].preprocess(batch)["attention_mask"].sum(dim=1).tolist()
        assert kept == [min(len(ids), MAX_SEQ_LEN) for ids in full]
        rows.extend(zip(map(len, before), map(len, full), kept, strict=True))
    return rows


def rate(count, total):
    return {"count": count, "denominator": total, "percent": 100 * count / total if total else None}


def length_summary(rows):
    keys = ("raw", "formatted_untruncated", "effective_nonpadding")
    return {key: {"over_512": rate(sum(n > MAX_SEQ_LEN for n in sizes), len(rows)),
                  "median": statistics.median(sizes), "max": max(sizes)}
            for key, sizes in zip(keys, zip(*rows, strict=True), strict=True)}


def ranking(hits):
    return sorted(hits, key=lambda did: -hits[did])


def funnel_summary(rows):
    top = [row for row in rows.values() if row["final_gold_rank"] is not None]
    parsed = [row for row in top if row["example_found"]]
    positive = [row for row in parsed if row["gold_passed"]]
    zero = [row for row in top if not row["passer_ids"]]
    n, k, p, g = len(rows), len(top), len(parsed), len(positive)
    passing = sum(len(row["passer_ids"]) for row in top)
    rank_deltas = [row["final_gold_rank"] - row["stage1_gold_rank"] for row in top]
    return {
        "gold_in_stage1_top50": rate(k, n),
        "gold_outside_stage1_top50": rate(n - k, n),
        "parseable_of_top50": rate(p, k), "parseable_of_sample": rate(p, n),
        "unparseable_of_top50": rate(k - p, k),
        "gold_pass_of_top50": rate(g, k), "gold_pass_of_parsed": rate(g, p),
        "gold_pass_of_sample": rate(g, n), "gold_fail_of_parsed": rate(p - g, p),
        "any_pass_of_top50": rate(k - len(zero), k),
        "any_pass_of_parsed": rate(k - len(zero), p),
        "zero_pass_of_top50": rate(len(zero), k),
        "zero_pass_of_parsed": rate(len(zero) - (k - p), p),
        "non_gold_pass_pairs": rate(passing - g, p * (DEFAULT_K - 1)),
        "passing_pairs": rate(passing, p * DEFAULT_K),
        "mean_passers_per_top50_query": passing / k,
        "mean_passers_per_parsed_query": passing / p,
        "final_gold_rank_top50_histogram": Counter(row["final_gold_rank"] for row in top),
        "final_gold_rank_top50_mean": statistics.mean(row["final_gold_rank"] for row in top),
        "final_gold_rank_top50_median": statistics.median(row["final_gold_rank"] for row in top),
        "gold_rank_among_passers_histogram": Counter(row["gold_rank_among_passers"] for row in positive),
        "gold_rank1_among_passers": rate(sum(row["gold_rank_among_passers"] == 1 for row in positive), g),
        "stage1_gold_top10_of_top50": rate(sum(row["stage1_gold_rank"] <= 10 for row in top), k),
        "final_gold_top10_of_top50": rate(sum(row["final_gold_rank"] <= 10 for row in top), k),
        "final_gold_top1_of_top50": rate(sum(row["final_gold_rank"] == 1 for row in top), k),
        "rank_movement_of_top50": {
            "improved": rate(sum(delta < 0 for delta in rank_deltas), k),
            "unchanged": rate(rank_deltas.count(0), k),
            "worse": rate(sum(delta > 0 for delta in rank_deltas), k),
        },
    }


def context_summary(queries, rows):
    import pyarrow.parquet as pq

    (path,) = SNAPSHOT.glob("queries/*.parquet")
    metadata = pq.read_table(path, columns=["_id", "meta_information"],
                             filters=[("partition", "=", "train")]).to_pylist()
    sources = {}
    for item in metadata:
        if (row := rows.get(item["_id"])) is None:
            continue
        url = item["meta_information"]["url"]
        host = urlsplit(url).netloc.lower().removeprefix("www.")
        counts = sources.setdefault(host, Counter())
        in_top50 = row["final_gold_rank"] is not None
        counts.update(sample=1, sample_parseable=row["example_found"], top50=in_top50,
                      top50_parseable=in_top50 and row["example_found"])

    missing = {qid: queries[qid] for qid, row in rows.items()
               if row["final_gold_rank"] is not None and not row["example_found"]}
    parsed = [row for row in rows.values() if row["example_found"]]
    parsed_in_top50 = sum(row["final_gold_rank"] is not None for row in parsed)
    assert sum(counts["sample"] for counts in sources.values()) == len(rows)
    truncation_groups = {
        "truncated": [row for row in rows.values() if row["query_tokens"][1] > MAX_SEQ_LEN],
        "not_truncated": [row for row in rows.values() if row["query_tokens"][1] <= MAX_SEQ_LEN],
    }
    return {
        "source_host_counts": dict(sorted(sources.items())),
        "parseable_of_sample": rate(len(parsed), len(rows)),
        "top50_of_parseable_sample": rate(parsed_in_top50, len(parsed)),
        "top50_of_unparseable_sample": rate(len(missing), len(rows) - len(parsed)),
        "skips_with_dashed_example_heading": rate(sum("-----Example" in text for text in missing.values()), len(missing)),
        "skips_matching_existing_colon_probe": rate(len(colon_candidates(missing)), len(missing)),
        "recall_by_query_truncation": {
            group: {cutoff: rate(sum((row["stage1_gold_rank"] or float("inf")) <= cutoff for row in members), len(members))
                    for cutoff in CUTOFFS}
            for group, members in truncation_groups.items()
        },
    }


def measure(output):
    import mteb
    import pyarrow.parquet as pq
    import torch
    from datasets import Dataset
    from mteb._create_dataloaders import _corpus_to_dict
    from mteb.models.search_wrappers import SearchEncoderWrapper
    from src.mteb_wrappers import E5Encoder, LiveExecReranker

    task = mteb.get_task("AppsRetrieval")
    assert task.metadata.dataset["revision"] == SNAPSHOT.name
    queries, train_docs, gold = load_train(SNAPSHOT)
    (corpus_path,) = SNAPSHOT.glob("corpus/*.parquet")
    corpus = Dataset(pq.read_table(corpus_path, columns=["_id", "text", "title"]).rename_columns(
        ["id", "text", "title"]))
    docs = dict(zip(corpus["id"], corpus["text"], strict=True))
    assert len(docs) == len(corpus) == 8765 and train_docs.keys() <= docs.keys()
    model_ref = Path.home() / ".cache/huggingface/hub/models--intfloat--e5-base-v2/refs/main"
    assert model_ref.read_text().strip() == MODEL_REVISION
    qids = random.Random(SEED).sample(sorted(queries), SAMPLE_SIZE)
    query_set = Dataset.from_dict({"id": qids, "text": [queries[qid] for qid in qids]})

    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    assert EMB_CACHE.is_file()
    initial_cache_hash = sha256(EMB_CACHE)
    e5 = E5(device="cpu")
    assert e5.model.default_prompt_name is None
    encoder = E5Encoder(e5, "prism/e5-base-v2-baseline")
    options = dict(task_metadata=task.metadata, hf_split="train", hf_subset="default",
                   encode_kwargs={"batch_size": e5.batch_size})
    search = SearchEncoderWrapper(encoder)
    search.index(corpus, **options)
    with patch.object(e5, "save"):
        stage1 = search.search(query_set, top_k=max(task.k_values), **options)
    orders = {qid: ranking(hits) for qid, hits in stage1.items()}
    assert all(len(order) == max(task.k_values) for order in orders.values())
    top50 = {qid: order[:DEFAULT_K] for qid, order in orders.items() if gold[qid] in order[:DEFAULT_K]}
    recall = {cutoff: rate(sum(gold[qid] in orders[qid][:cutoff] for qid in qids), len(qids))
              for cutoff in CUTOFFS}
    print("Stage-1 strict recall:", recall, flush=True)

    query_order = sorted(queries)
    query_lengths = dict(zip(query_order, token_lengths(
        e5.model, [queries[qid] for qid in query_order], [QUERY_PREFIX + queries[qid] for qid in query_order]),
        strict=True))
    doc_lengths = dict(zip(docs, token_lengths(
        e5.model, list(docs.values()), [DOC_PREFIX + _corpus_to_dict(row)["text"] for row in corpus]),
        strict=True))
    lengths = {"sampled_train_queries": length_summary([query_lengths[qid] for qid in qids]),
               "all_train_queries": length_summary(list(query_lengths.values())),
               "indexed_corpus": length_summary(list(doc_lengths.values())),
               "train_corpus_subset": length_summary([doc_lengths[did] for did in train_docs])}
    print("Token lengths:", lengths, flush=True)

    # Reuse the retrieved scores and run the same live wrapper used by evaluate.py.
    hits = {qid: stage1[qid].copy() for qid in top50}
    batch = {"id": list(top50), "text": [queries[qid] for qid in top50]}
    with TemporaryDirectory(prefix="ctrlfind_funnel_") as scratch:
        checkpoint = Path(scratch) / "verification.jsonl"
        reranker = LiveExecReranker(encoder, checkpoint)
        reranker.doc_text = docs
        with patch.object(SearchEncoderWrapper, "search", return_value=hits):
            final = reranker.search(batch, top_k=max(task.k_values), **options)
        verdicts = {row["qid"]: row for line in checkpoint.read_text(encoding="utf-8").splitlines()
                    if line.strip() for row in [json.loads(line)]}

    rows = {}
    for qid in qids:
        did, order = gold[qid], orders[qid]
        initial_rank = order.index(did) + 1 if did in order else None
        parsed = execute.parse_example(queries[qid]) is not None
        row = {"gold_id": did, "query_tokens": query_lengths[qid], "gold_tokens": doc_lengths[did],
               "stage1_gold_rank": initial_rank, "stage1_top50": order[:DEFAULT_K],
               "example_found": parsed, "gold_passed": None, "passer_ids": [], "final_gold_rank": None,
               "gold_rank_among_passers": None}
        if qid in top50:
            assert (qid in verdicts) == parsed
            record = verdicts.get(qid, {})
            passed = set(record.get("passer_ids", []))
            final_order = ranking(final[qid])
            ordered_passers = [candidate for candidate in final_order if candidate in passed]
            assert passed <= set(top50[qid]) and final_order[:len(passed)] == ordered_passers
            row.update(gold_passed=(did in passed) if parsed else None,
                       passer_ids=ordered_passers, final_gold_rank=final_order.index(did) + 1,
                       gold_rank_among_passers=ordered_passers.index(did) + 1 if did in passed else None,
                       verification_key=record.get("verification_key"),
                       allow_trailing_padding=record.get("allow_trailing_padding"))
        rows[qid] = row

    assert sha256(EMB_CACHE) == initial_cache_hash
    result = {
        "dataset": DATASET, "dataset_revision": SNAPSHOT.name, "query_split": "train",
        "sampling": {"method": "random.Random(seed).sample(sorted(training_query_ids), 500)",
                     "seed": SEED, "universe": len(queries), "query_ids": qids},
        "model": {"name": MODEL_NAME, "revision": MODEL_REVISION, "max_seq_length": MAX_SEQ_LEN,
                  "query_prefix": QUERY_PREFIX, "document_prefix": DOC_PREFIX, "device": e5.device,
                  "normalize_embeddings": True, "batch_size": e5.batch_size, "torch_threads": THREADS,
                  "torch_interop_threads": 1, "similarity": "MTEB cosine", "search_depth": max(task.k_values),
                  "embedding_cache_sha256": initial_cache_hash},
        "corpus": {"count": len(corpus), "train_count": len(train_docs), "parquet_sha256": sha256(corpus_path),
                   "ordered_doc_ids_sha256": hashlib.sha256("\n".join(docs).encode()).hexdigest()},
        "verification": {"k": DEFAULT_K, "timeout_seconds": EXEC_TIMEOUT, "workers": os.cpu_count() or 2,
                         "checkpoint": "fresh, process-local scratch", "verifier_sha256": sha256(Path(execute.__file__)),
                         "windows_job_sha256": sha256(ROOT / "src/_windows_job.py"),
                         "reranker_sha256": sha256(ROOT / "src/mteb_wrappers.py")},
        "environment": {"python": sys.version, "platform": platform.platform(), "utf8_mode": sys.flags.utf8_mode,
                        "packages": {name: importlib.metadata.version(name) for name in
                                     ("mteb", "sentence-transformers", "transformers", "tokenizers", "torch", "pyarrow", "numpy", "datasets")}},
        "recall": recall, "token_lengths": lengths, "funnel": funnel_summary(rows),
        "context": context_summary(queries, rows), "rows": rows,
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, separators=(",", ":"))
        handle.write("\n")
    print("Funnel:", result["funnel"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    measure(args.output)
