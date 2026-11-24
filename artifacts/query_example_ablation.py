"""Train-only E5 ablation with fixed populations and immutable evaluation inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from time import perf_counter

from verification_research import SNAPSHOT, load_train
from src.cache import EmbeddingCache
from src.config import DEFAULT_K, DOC_PREFIX, EMB_CACHE, MAX_SEQ_LEN, MODEL_NAME
from src.encoder import E5

MODEL_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
SEED = 20261125
UNIFORM_COUNT = 500
THREADS = 4
BATCH = 16
SAVED_PANEL = Path(__file__).with_name("verification_performance_train.json")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, separators=(",", ":"), ensure_ascii=True)
        handle.write("\n")


def workload():
    queries, docs, gold = load_train(SNAPSHOT)
    saved = read_json(SAVED_PANEL)
    assert saved["revision"] == SNAPSHOT.name and saved["k"] == DEFAULT_K
    uniform = random.Random(SEED).sample(sorted(queries), UNIFORM_COUNT)
    verification = list(saved["plan"])
    populations = {"uniform": uniform, "saved_verification": verification}
    return queries, docs, gold, populations


def model_and_corpus(docs):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    model_path = Path.home() / ".cache/huggingface/hub/models--intfloat--e5-base-v2/snapshots" / MODEL_REVISION
    e5 = E5.__new__(E5)
    e5.device, e5.batch_size = "cpu", BATCH
    e5.model = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
    e5.model.max_seq_length = MAX_SEQ_LEN
    # A nonexistent research cache path starts cold. The runner never calls save.
    e5.cache = EmbeddingCache(Path(os.devnull).with_name("ctrlfind-ablation-unused.npz"), MODEL_NAME)
    assert len(e5.cache) == 0
    ids = sorted(docs)
    doc_cache = EmbeddingCache(EMB_CACHE, MODEL_NAME)

    def missing(texts):
        raise RuntimeError(f"{len(texts)} missing training documents in the reference embedding cache")

    vectors, misses = doc_cache.get_many([DOC_PREFIX + docs[did] for did in ids], missing)
    assert misses == 0
    return e5, ids, vectors


def measure(e5, doc_ids, vectors, gold, qids, texts):
    import numpy as np

    positions = {did: i for i, did in enumerate(doc_ids)}
    rows, batches = {}, []
    for start in range(0, len(qids), BATCH):
        ids, queries = qids[start:start + BATCH], texts[start:start + BATCH]
        t0 = perf_counter()
        qvec, misses = e5.encode_queries(queries)
        t1 = perf_counter()
        orders = np.argsort(-(qvec @ vectors.T), axis=1, kind="stable")
        t2 = perf_counter()
        for qid, order in zip(ids, orders):
            pos = positions[gold[qid]]
            rows[qid] = {
                "rank": int(np.flatnonzero(order == pos)[0]) + 1,
                "top": [doc_ids[i] for i in order[:DEFAULT_K]],
            }
        batches.append({"ids": ids, "misses": misses, "encode": t1 - t0, "rank": t2 - t1,
                        "total": perf_counter() - t0})
        if (start // BATCH + 1) % 5 == 0 or start + BATCH >= len(qids):
            print(f"{len(rows)}/{len(qids)} scored", flush=True)
    return rows, batches


def baseline(output):
    queries, docs, gold, populations = workload()
    qids = list(dict.fromkeys(qid for ids in populations.values() for qid in ids))
    e5, ids, vectors = model_and_corpus(docs)
    rows, batches = measure(e5, ids, vectors, gold, qids, [queries[qid] for qid in qids])
    write_json(output, {
        "train_revision": SNAPSHOT.name, "model_revision": MODEL_REVISION, "seed": SEED,
        "k": DEFAULT_K, "threads": THREADS, "batch_size": BATCH, "populations": populations,
        "corpus_count": len(ids), "corpus_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "rows": rows, "batches": batches,
    })



def rank_metrics(ranks):
    import math
    n = len(ranks)
    return {
        "recall_at_10": 100 * sum(r <= 10 for r in ranks) / n,
        "recall_at_50": 100 * sum(r <= DEFAULT_K for r in ranks) / n,
        "ndcg_at_10": 100 * sum(1 / math.log2(r + 1) for r in ranks if r <= 10) / n,
        "mrr_at_10": 100 * sum(1 / r for r in ranks if r <= 10) / n,
    }


def compare(qids, before, after, edits):
    import numpy as np

    old = np.asarray([before[qid]["rank"] for qid in qids])
    new = np.asarray([after[qid]["rank"] for qid in qids])
    changes = (new <= DEFAULT_K).astype(int) - (old <= DEFAULT_K)
    draw = np.random.default_rng(SEED).integers(0, len(old), size=(5000, len(old)))
    return {
        "count": len(qids), "edited": sum(qid in edits for qid in qids),
        "pre": rank_metrics(old), "post": rank_metrics(new),
        "improved": int(sum(new < old)), "degraded": int(sum(new > old)),
        "unchanged": int(sum(new == old)),
        "entered_at_10": int(sum((old > 10) & (new <= 10))),
        "left_at_10": int(sum((old <= 10) & (new > 10))),
        "entered_at_50": int(sum((old > DEFAULT_K) & (new <= DEFAULT_K))),
        "left_at_50": int(sum((old <= DEFAULT_K) & (new > DEFAULT_K))),
        "recall50_paired_95_ci_pp": (100 * np.quantile(changes[draw].mean(axis=1), [.025, .975])).tolist(),
    }


def experiment(output, baseline_path):
    import tempfile
    import numpy as np

    from query_example_preprocessing import strip_examples
    from src.config import QUERY_PREFIX
    from src.execute import parse_example

    reference = read_json(baseline_path)
    queries, docs, gold, populations = workload()
    assert reference["populations"] == populations
    assert reference["train_revision"] == SNAPSHOT.name and reference["model_revision"] == MODEL_REVISION
    assert (reference["k"], reference["threads"], reference["batch_size"]) == (DEFAULT_K, THREADS, BATCH)
    before = reference["rows"]
    t0 = perf_counter()
    census = {qid: text for qid, query in queries.items() if (text := strip_examples(query)) != query}
    census_seconds = perf_counter() - t0
    edits = {qid: census[qid] for qid in before if qid in census}
    affected = list(edits)
    e5, doc_ids, vectors = model_and_corpus(docs)
    assert reference["corpus_sha256"] == hashlib.sha256("\n".join(doc_ids).encode()).hexdigest()

    # Equal batch shapes in both cold arms. Unchanged statements reuse exact baseline ranks.
    with tempfile.TemporaryDirectory(prefix="ctrlfind_query_ablation_") as directory:
        scratch = Path(directory)
        paired, timing = {}, {}
        for tag, source in (("original", queries), ("stripped", edits)):
            e5.cache = EmbeddingCache(scratch / f"{tag}.npz", MODEL_NAME)
            paired[tag], timing[tag] = measure(e5, doc_ids, vectors, gold, affected, [source[q] for q in affected])
        new = {**before, **paired["stripped"]}

        # Observe actual E5 vectors through the same content-addressed cache.
        qid = next(q for q in affected if e5.model.tokenize([QUERY_PREFIX + queries[q]])["input_ids"][0].tolist()
                   != e5.model.tokenize([QUERY_PREFIX + edits[q]])["input_ids"][0].tolist())
        texts = [queries[qid], edits[qid]]
        e5.cache = EmbeddingCache(scratch / "identity.npz", MODEL_NAME)
        first, misses = e5.encode_queries(texts)
        again, warm_misses = e5.encode_queries(texts[::-1])
        assert misses == 2 and warm_misses == 0 and np.array_equal(first, again[::-1])
        assert not np.array_equal(first[0], first[1])

        # Warm the model once, then interleave single-query cold retrieval on fixed edits.
        latency_ids = random.Random(SEED + 1).sample(sorted(affected), 24)
        latency = []
        for qid in latency_ids:
            for tag in (("original", "stripped") if len(latency) % 4 == 0 else ("stripped", "original")):
                e5.cache = EmbeddingCache(scratch / "unwritten_latency.npz", MODEL_NAME)
                start = perf_counter()
                text = strip_examples(queries[qid]) if tag == "stripped" else queries[qid]
                encoded, _ = e5.encode_queries([text])
                np.argsort(-(encoded @ vectors.T), axis=1, kind="stable")[0, :DEFAULT_K]
                latency.append({"qid": qid, "arm": tag, "seconds": perf_counter() - start})

    token_diagnostics = {}
    for qid in affected:
        pair = e5.model.tokenize([QUERY_PREFIX + queries[qid], QUERY_PREFIX + edits[qid]])
        ids = [row[mask.bool()].tolist() for row, mask in zip(pair["input_ids"], pair["attention_mask"])]
        token_diagnostics[qid] = {
            "effective_tokens": [len(row) for row in ids], "identical": ids[0] == ids[1],
            "removed_characters": len(queries[qid]) - len(edits[qid]),
            "source_sha256": hashlib.sha256(queries[qid].encode()).hexdigest(),
            "variant_sha256": hashlib.sha256(edits[qid].encode()).hexdigest(),
        }
    report = {
        "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "training_coverage": {"total": len(queries), "edited": len(census),
                              "edited_with_existing_example": sum(parse_example(queries[q]) is not None for q in census)},
        "census_seconds": census_seconds,
        "populations": {**{tag: compare(ids, before, new, edits) for tag, ids in populations.items()},
                        "edited_union": compare(affected, before, new, edits)},
        "paired_original_changed_ranks": [q for q in affected if paired["original"][q]["rank"] != before[q]["rank"]],
        "paired_original_changed_top50": [q for q in affected if paired["original"][q]["top"] != before[q]["top"]],
        "reference_rows_for_edits": paired["original"], "rows": paired["stripped"],
        "edit_details": token_diagnostics, "batches": timing, "single_query_cold": latency,
        "cache_check": {"separate_misses": misses, "repeat_misses": warm_misses, "vectors_distinct": True},
    }
    write_json(output, report)
    print(json.dumps(report["populations"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("baseline", "experiment"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OPENBLAS_NUM_THREADS=str(THREADS),
                      OMP_NUM_THREADS=str(THREADS), MKL_NUM_THREADS=str(THREADS))
    if args.mode == "baseline":
        baseline(args.output)
    elif args.baseline is None:
        parser.error("experiment requires --baseline")
    else:
        experiment(args.output, args.baseline)
