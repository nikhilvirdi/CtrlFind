"""Evaluate the amended V2 latency gate, with a single guarded B run if it passes."""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import random
import time

import numpy as np
from mteb.similarity_functions import cos_sim

import retrieval_consistency as audit
from dev_stdin_full_baseline import VECTORS
from dev_stdin_query_repr import INPUTS, TokenEncoder, collapse, frozen_shape_cosine, summarize
from src.config import MAX_SEQ_LEN, QUERY_PREFIX


AMENDMENT = "### Query representation latency amendment"
A_RESULT = audit.HERE / "dev_stdin_query_repr_run" / "A_results.json"
COUNTS = {"truncated": 105, "not_truncated": 119}
LIMIT = 0.3


def amendment_identity():
    text = (audit.ROOT / "docs" / "research-log.md").read_text(encoding="utf-8")
    section = text.split(AMENDMENT + "\n", 1)[1].split("\n### ", 1)[0].strip()
    assert "2026-11-25 03:01:52 +05:30" in section and "<= 0.3 s/query" in section
    return {"section": AMENDMENT, "sha256": audit.sha_bytes(section.encode("utf-8"))}


def test_token_counts(output):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(audit.MODEL, local_files_only=True)
    assert tokenizer.truncation_side == "right"
    path = audit.parquet("queries/*.parquet")
    train, = audit.pq.read_table(
        path, columns=["text"], filters=[("partition", "=", "train"), ("_id", "=", "q632")]
    ).to_pylist()
    census = audit.read_json(INPUTS[2])
    probe = tokenizer.encode(QUERY_PREFIX + train["text"], add_special_tokens=True, truncation=False, verbose=False)
    assert len(probe) == census["rows"]["q632"]["formatted_tokens"]
    # No file containing test relevance labels is opened by this mode.
    data = audit.pq.read_table(path, columns=["text"], filters=[("partition", "=", "test")])
    assert data.num_rows == 3765
    counts = Counter()
    for batch in data.to_batches(max_chunksize=64):
        inputs = [QUERY_PREFIX + text for text in batch["text"].to_pylist()]
        encoded = tokenizer(inputs, add_special_tokens=True, truncation=False, verbose=False)["input_ids"]
        counts.update(map(len, encoded))
    over = sum(count for length, count in counts.items() if length > MAX_SEQ_LEN)
    assert sum(counts.values()) == data.num_rows
    result = {"amendment": amendment_identity(), "dataset_revision": audit.REVISION,
              "model_revision": audit.MODEL_REVISION, "partition": "test", "columns_read": ["text"],
              "query_prefix": QUERY_PREFIX, "special_tokens_included": True, "cap": MAX_SEQ_LEN,
              "over_cap": audit.rate(over, data.num_rows), "token_counts_histogram": dict(sorted(counts.items()))}
    audit.write_json(output / "test_token_counts.json", result)
    print("Label-free token count", result["over_cap"], flush=True)


def latency_sample(split, census):
    source = census["rows"]
    a_ids = split["halves"]["A"]["query_ids"]
    groups = {name: sorted(qid for qid in a_ids
                          if (source[qid]["formatted_tokens"] > MAX_SEQ_LEN) == (name == "truncated"))
              for name in COUNTS}
    sample, orders = [], {}
    for name, population in groups.items():
        draw = random.Random(f"{audit.SEED}:latency-amendment:{name}").sample(population, COUNTS[name])
        for i, qid in enumerate(draw):
            orders[qid] = ("C0", "V2") if i % 2 == 0 else ("V2", "C0")
        sample.extend(draw)
    random.Random(f"{audit.SEED}:latency-amendment:mix").shuffle(sample)
    used = set(sample)
    short = [qid for qid in groups["not_truncated"] if qid not in used]
    long = [qid for qid in groups["truncated"] if qid not in used]
    warmup = [short[0], long[0], max(long, key=lambda qid: source[qid]["formatted_tokens"])]
    assert len(warmup) == len(set(warmup)) == 3 and not set(warmup) & used
    return sample, orders, warmup


def timed_query(arm, query, encoder, docs):
    start = time.perf_counter()
    if arm == "C0":
        vectors = encoder.e5._encode_raw([QUERY_PREFIX + query])
    else:
        body = encoder.body(query)
        if len(body) > encoder.budget:
            _, _, sequences = encoder.plan([body], "V2")
        else:
            sequences = [encoder.window(body)]
        # The inherited encoder makes one forward call for inputs within a single batch.
        assert 1 <= len(sequences) <= audit.BATCH
        vectors = encoder.embed(sequences)
    encoded = time.perf_counter()
    scores = cos_sim(vectors, docs).numpy().max(axis=0)
    elapsed = time.perf_counter() - start
    assert scores.shape == (len(docs),) and np.isfinite(scores).all()
    return {"query_side_seconds": elapsed, "format_encode_seconds": encoded - start, "windows": len(vectors)}


def distribution(values):
    return {"mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95))}


def latency(output):
    registered = amendment_identity()
    assert not (output / "B_started.json").exists() and not (output / "latency.json").exists()
    frozen = {path.name: audit.sha_file(path) for path in audit.FROZEN}
    split, baseline, census = map(audit.read_json, INPUTS)
    sample, orders, warmup = latency_sample(split, census)
    assert set(sample + warmup).isdisjoint(split["halves"]["B"]["query_ids"])
    query_rows = audit.pq.read_table(
        audit.parquet("queries/*.parquet"), columns=["_id", "text"],
        filters=[("partition", "=", "train"), ("_id", "in", sample + warmup)]
    ).to_pylist()
    query = {row["_id"]: row["text"] for row in query_rows}
    assert set(query) == set(sample + warmup)
    train_doc_ids = census["corpus"]["document_ids"]
    # Only partition membership is needed to validate the corpus for timing.
    ids, _, _, _, _, passages, _, _ = audit.corpus(
        {did: did for did in train_doc_ids}, audit.read_json(audit.ALL_LEDGER), audit.read_json(audit.TRAIN_LEDGER))
    assert ids == baseline["config"]["candidate_corpus"]["document_ids_in_parquet_order"]
    assert audit.vector_digest(passages) == baseline["config"]["candidate_corpus"]["vector_sha256"]
    encoder = TokenEncoder(audit.load_model(output / "unused.npz"))
    for qid in warmup:
        for arm in ("C0", "V2"):
            timed_query(arm, query[qid], encoder, passages)
    print("Three warmup queries finished. Starting 224 serial timing pairs.", flush=True)
    rows = {}
    for i, qid in enumerate(sample, 1):
        meta = census["rows"][qid]
        measured = {arm: timed_query(arm, query[qid], encoder, passages) for arm in orders[qid]}
        rows[qid] = {"formatted_tokens": meta["formatted_tokens"], "source": meta["source"],
                     "truncated": meta["formatted_tokens"] > MAX_SEQ_LEN, "order": orders[qid], **measured}
        assert (measured["V2"]["windows"] > 1) == rows[qid]["truncated"]
        if i % 16 == 0:
            print(f"Timed {i}/{len(sample)} query pairs", flush=True)

    by_group, population, amortized = {}, {}, {}
    for name in COUNTS:
        truncated = name == "truncated"
        group = [row for row in rows.values() if row["truncated"] == truncated]
        assert len(group) == COUNTS[name]
        population[name] = sum((row["formatted_tokens"] > MAX_SEQ_LEN) == truncated for row in census["rows"].values())
        by_group[name] = {arm: {metric: distribution([d[arm][metric] for d in group])
                               for metric in ("query_side_seconds", "format_encode_seconds")}
                          for arm in ("C0", "V2")}
    assert population == {"truncated": 818, "not_truncated": 928}
    for arm in ("C0", "V2"):
        amortized[arm] = sum(population[name] * by_group[name][arm]["query_side_seconds"]["mean"]
                             for name in COUNTS) / sum(population.values())
    added = amortized["V2"] - amortized["C0"]
    assert frozen == {path.name: audit.sha_file(path) for path in audit.FROZEN}
    result = {"amendment": registered, "input_sha256": {p.name: audit.sha_file(p) for p in INPUTS},
              "original_A_sha256": audit.sha_file(A_RESULT), "sample_seed": audit.SEED, "sample_sizes": COUNTS,
              "warmup_ids_discarded": warmup, "ordered_query_ids": sample, "census_weights": population,
              "runtime": {"threads": audit.THREADS, "interop_threads": 1, "logical_processors": os.cpu_count(),
                          "model_revision": audit.MODEL_REVISION, "candidate_vectors": len(passages)},
              "by_group": by_group, "amortized_seconds_per_query": amortized, "amortized_added_seconds": added,
              "limit_seconds": LIMIT, "passed": added <= LIMIT, "rows": rows, "frozen_inputs": frozen}
    audit.write_json(output / "latency.json", result)
    print("Amortized query-side seconds", amortized, "added", added, "pass", result["passed"], flush=True)


def evaluate_b(output):
    registered = amendment_identity()
    frozen = {path.name: audit.sha_file(path) for path in audit.FROZEN}
    timing = audit.read_json(output / "latency.json")
    assert timing["amendment"] == registered and timing["passed"] and timing["amortized_added_seconds"] <= LIMIT
    assert timing["input_sha256"] == {p.name: audit.sha_file(p) for p in INPUTS}
    assert timing["original_A_sha256"] == audit.sha_file(A_RESULT)
    a_results = audit.read_json(A_RESULT)
    assert all(a_results["arms"]["V2"]["measures"]["gates_excluding_latency"].values())
    lock = {"arm": "V2", "amendment": registered, "A_sha256": audit.sha_file(A_RESULT),
            "latency_sha256": audit.sha_file(output / "latency.json")}
    audit.write_json(output / "B_started.json", lock)

    split, baseline, census = map(audit.read_json, INPUTS)
    all_ids = split["halves"]["B"]["query_ids"]
    selected = [qid for qid in all_ids if census["rows"][qid]["formatted_tokens"] > MAX_SEQ_LEN]
    queries, sources, gold, _, dev, old, ablation = audit.inputs()
    ids, _, _, _, _, passages, _, _ = audit.corpus(gold, old, ablation)
    assert ids == baseline["config"]["candidate_corpus"]["document_ids_in_parquet_order"]
    assert audit.vector_digest(passages) == baseline["config"]["candidate_corpus"]["vector_sha256"]
    c0_vectors, manifest = audit.read_vectors(VECTORS, dev, queries)
    assert manifest == baseline["config"]["query_vector_manifest"]
    positions = [dev.index(qid) for qid in selected]
    c0 = frozen_shape_cosine(c0_vectors[positions], passages, len(dev), positions)
    original = baseline["rows"]
    replay = audit.fixed_ranks(c0, ids, selected, gold)
    assert all(row["rank"] == original[qid]["rank"] and row["top200"] == original[qid]["top200"]
               for qid, row in replay.items())

    encoder = TokenEncoder(audit.load_model(output / "unused.npz"))
    parts = [encoder.body(queries[qid]) for qid in selected]
    assert all(len(body) + len(encoder.prefix) + encoder.special == census["rows"][qid]["formatted_tokens"]
               for qid, body in zip(selected, parts, strict=True))
    owners, offsets, inputs = encoder.plan(parts, "V2", skip_first=True)
    embeddings = encoder.embed(inputs, "B locked V2")
    with (output / "B_V2_vectors.npz").open("xb") as out:
        np.savez_compressed(out, query_ids=selected, owners=owners, offsets=offsets, vectors=embeddings)
    window_scores = frozen_shape_cosine(embeddings, passages, len(dev), list(range(len(embeddings))))
    scores = np.maximum(c0, collapse(window_scores, owners, len(selected)))
    rows = audit.fixed_ranks(scores, ids, selected, gold)
    for qid, row in rows.items():
        row.update(C0_rank=original[qid]["rank"], source=sources[qid])
    measures = summarize("B", all_ids, selected, sources, original, rows)
    decision = "KEEP" if all(measures["gates_excluding_latency"].values()) else "REJECTED"
    assert frozen == {path.name: audit.sha_file(path) for path in audit.FROZEN}
    audit.write_json(output / "B_results.json", {"lock": lock, "query_ids": all_ids, "truncated": selected,
                                                "measures": measures, "decision": decision, "rows": rows})
    print("B", decision, measures["gates_excluding_latency"], flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("tokens", "latency", "B"))
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    audit.configure()
    args.output.mkdir(parents=True, exist_ok=True)
    {"tokens": test_token_counts, "latency": latency, "B": evaluate_b}[args.mode](args.output)
