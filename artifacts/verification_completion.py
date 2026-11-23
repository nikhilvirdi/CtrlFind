"""Capture or summarize the fixed, training-only verification replay."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

from verification_research import SNAPSHOT, WORKERS, decode, load_train, sample
from src import execute
from src.config import DEFAULT_K, EXEC_TIMEOUT

RANK_COUNT = 40
RANK_SEED = 20261129


def plan_ranks(queries, docs, gold, eligible):
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    from src.cache import EmbeddingCache
    from src.config import DOC_PREFIX, EMB_CACHE, MAX_SEQ_LEN, MODEL_NAME, QUERY_PREFIX

    def missing(texts):
        raise ValueError(f"missing {len(texts)} cached training document embeddings")

    ids = sorted(docs)
    cache = EmbeddingCache(EMB_CACHE, MODEL_NAME)
    docs_vec, _ = cache.get_many([DOC_PREFIX + docs[did] for did in ids], missing)
    chosen = random.Random(RANK_SEED).sample(sorted(eligible), RANK_COUNT)
    torch.set_num_threads(4)
    model = SentenceTransformer(MODEL_NAME, device="cpu", local_files_only=True)
    model.max_seq_length = MAX_SEQ_LEN
    rank_plan = {}
    for start in range(0, len(chosen), 8):
        qids = chosen[start:start + 8]
        vec = model.encode(
            [QUERY_PREFIX + queries[qid] for qid in qids], batch_size=8,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        )
        orders = np.argsort(-(vec @ docs_vec.T), axis=1, kind="stable")
        for qid, order in zip(qids, orders):
            ranked = [ids[i] for i in order]
            rank_plan[qid] = {"top": ranked[:DEFAULT_K], "r": ranked.index(gold[qid]) + 1}
    return rank_plan


def digest(rows):
    items = [(row["job"], row["candidates"]) for row in rows]
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


def phase(obs):
    if not obs or obs.get("exception"):
        return "launch_error"
    if obs.get("timeout"):
        return "timeout"
    if obs["returncode"] != 0:
        return "nonzero"
    return "complete" if isinstance(obs["stdout"], str) else "decode_error"


class CandidateSource(str):
    """Give byte-identical snippets distinct observation identities."""


def observe(sources, example):
    observations = {}
    active = threading.local()
    runner = execute.run_snippet

    def run_snippet(code, *args):
        active.record = observations[id(code)] = {}
        return runner(code, *args)

    def run(args, **kwargs):
        start = time.perf_counter()
        obs = active.record
        try:
            result = subprocess.run(args, **kwargs)
            obs.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            return result
        except subprocess.TimeoutExpired as exc:
            obs.update(timeout=True, stdout=decode(exc.stdout), stderr=decode(exc.stderr))
            raise
        except Exception as exc:
            obs["exception"] = repr(exc)
            raise
        finally:
            obs["seconds"] = round(time.perf_counter() - start, 4)

    with (
        patch.object(execute, "subprocess", SimpleNamespace(run=run)),
        patch.object(execute, "run_snippet", side_effect=run_snippet),
    ):
        exact = execute.verify(sources, example, timeout=EXEC_TIMEOUT, workers=WORKERS)
    return exact, [observations.get(id(source), {}) for source in sources]


def run_job(qid, candidates, initial_rank, queries, docs, gold, example):
    start = time.perf_counter()
    source = [CandidateSource(docs[did]) for did in candidates]
    exact, observed = observe(source, example)
    raw = {id(code): obs["stdout"].strip() if phase(obs) == "complete" else None
           for code, obs in zip(source, observed)}
    with patch.object(execute, "run_snippet", side_effect=lambda code, *args: raw[id(code)]):
        scoped = execute.verify(
            source, example, timeout=EXEC_TIMEOUT, workers=WORKERS,
            allow_trailing_padding=execute.allows_trailing_padding(queries[qid]),
        )
    detail = None
    if gold[qid] in candidates:
        obs = observed[candidates.index(gold[qid])]
        detail = {"stdout": (obs.get("stdout") or "")[:500], "stderr": (obs.get("stderr") or "")[-400:],
                  "exp": example[1][:500]}
    return {
        "job": qid, "gold": gold[qid], "candidates": candidates, "initial_rank": initial_rank,
        "exact": [did for did, ok in zip(candidates, exact) if ok],
        "base": [did for did, ok in zip(candidates, scoped) if ok],
        "outcomes": {did: {"phase": phase(obs), "seconds": obs.get("seconds")}
                     for did, obs in zip(candidates, observed)},
        "gold_detail": detail, "elapsed": round(time.perf_counter() - start, 4),
    }


def capture(path, jobs, queries, docs, gold, examples):
    with path.open("x", encoding="utf-8") as handle:
        for count, (qid, ids, rank) in enumerate(jobs, 1):
            row = run_job(qid, ids, rank, queries, docs, gold, examples[qid])
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if count % 20 == 0:
                print(path.name, count, flush=True)


def summary(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    latencies = sorted(row["elapsed"] for row in rows)
    positive = sum(row["gold"] in row["candidates"] for row in rows)
    pairs = sum(len(row["candidates"]) for row in rows)
    print(path, len(rows), digest(rows))
    print("attempts", pairs, Counter(o["phase"] for row in rows for o in row["outcomes"].values()))
    print("seconds total/median/p95/max", sum(latencies), statistics.median(latencies),
          latencies[math.ceil(.95 * len(rows)) - 1], latencies[-1])
    for policy in ("cosine", "exact", "base"):
        if policy != "cosine":
            pos = sum(row["gold"] in row[policy] for row in rows)
            wrong = sum(d != row["gold"] for row in rows for d in row[policy])
            print(policy, f"gold {pos}/{positive}, non-gold {wrong}/{pairs - positive}")
        if rows[0].get("initial_rank") is not None:
            ranks = []
            for row in rows:
                ranked = sorted(row["candidates"], key=lambda did: did not in row.get(policy, []))
                rank = ranked.index(row["gold"]) + 1 if row["gold"] in ranked else row["initial_rank"]
                ranks.append(rank)
            print(policy, "NDCG@10, MRR@10", 100 * sum(1 / math.log2(r + 1) for r in ranks if r <= 10) / len(ranks),
                  100 * sum(1 / r for r in ranks if r <= 10) / len(ranks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "capture", "summary"))
    parser.add_argument("paths", type=Path, nargs="+",
                        help="plan: PLAN. capture: PLAN PANEL RANK. summary: one or more JSONL files.")
    args = parser.parse_args()
    if args.command == "summary":
        for path in args.paths:
            summary(path)
        return
    if len(args.paths) != (1 if args.command == "plan" else 3):
        parser.error("use PLAN for plan, or PLAN PANEL RANK for capture")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    queries, docs, gold = load_train(SNAPSHOT)
    examples = {qid: ex for qid, text in queries.items() if (ex := execute.parse_example(text)) is not None}
    assert len(examples) == 511
    print(f"coverage {len(examples)}/{len(queries)} on training revision {SNAPSHOT.name}")
    if args.command == "plan":
        with args.paths[0].open("x", encoding="utf-8") as handle:
            json.dump(plan_ranks(queries, docs, gold, examples), handle)
        return
    rank_plan = json.loads(args.paths[0].read_text(encoding="utf-8"))
    chosen = random.Random(RANK_SEED).sample(sorted(examples), RANK_COUNT)
    assert list(rank_plan) == chosen
    assert all(len(v["top"]) == DEFAULT_K and set(v["top"]) <= docs.keys() for v in rank_plan.values())
    panel = [(qid, [did, *wrong], None) for qid, did, ex, wrong in sample(queries, docs, gold) if ex is not None]
    rank = [(qid, entry["top"], entry["r"]) for qid, entry in rank_plan.items()]
    capture(args.paths[1], panel, queries, docs, gold, examples)
    capture(args.paths[2], rank, queries, docs, gold, examples)


if __name__ == "__main__":
    main()
