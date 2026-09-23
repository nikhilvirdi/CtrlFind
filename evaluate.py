"""Produce the MTEB results file for AppsRetrieval.

  python evaluate.py --mode cached     stage 2 verdicts from artifacts/exec_results.jsonl
  python evaluate.py --mode full       run stage 2 live (hours on CPU, resumable)
  python evaluate.py --mode baseline   stage 1 only

Writes artifacts/appsretrieval_results.json.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from src.config import (ARTIFACTS, DEFAULT_K, DOC_IDS, EXEC_RESULTS,
                        RESULTS_JSON)


def load_cached_verdicts() -> dict[str, set[str]]:
    """Read stage 2 verdicts. Accepts both record formats:
    {"qid", "passer_ids": [doc ids]} and the original {"qid", "passers": [positions]}.
    """
    doc_ids = np.load(DOC_IDS, allow_pickle=True) if DOC_IDS.exists() else None
    out: dict[str, set[str]] = {}
    for line in EXEC_RESULTS.open(encoding="utf-8"):
        rec = json.loads(line)
        if "passer_ids" in rec:
            ids = rec["passer_ids"]
        else:
            if doc_ids is None:
                raise SystemExit(f"{DOC_IDS} is needed to read position-based results")
            ids = [str(doc_ids[i]) for i in rec["passers"]]
        if ids:
            out[rec["qid"]] = set(ids)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cached", "full", "baseline"], default="cached")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    args = ap.parse_args()

    import mteb
    from src.encoder import E5
    from src.mteb_wrappers import (CachedExecReranker, E5Encoder,
                                   LiveExecReranker)

    e5 = E5()
    print(f"device: {e5.device}, cached embeddings: {len(e5.cache)}")

    if args.mode == "baseline":
        model = E5Encoder(e5, "prism/e5-base-v2-baseline")
    elif args.mode == "cached":
        verdicts = load_cached_verdicts()
        print(f"verdicts loaded: {len(verdicts)} queries with passing snippets")
        model = CachedExecReranker(E5Encoder(e5, "prism/e5-base-v2-exec-rerank"), verdicts)
    else:
        model = LiveExecReranker(E5Encoder(e5, "prism/e5-base-v2-exec-rerank-live"),
                                 ARTIFACTS / "exec_results_live.jsonl", k=args.k)

    t0 = time.time()
    result = mteb.evaluate(model, [mteb.get_task("AppsRetrieval")],
                           encode_kwargs={"batch_size": e5.batch_size})
    elapsed = time.time() - t0

    out = list(result.task_results)[0].to_dict()
    s = out["scores"]["test"][0]
    print(f"\nNDCG@10 {s['ndcg_at_10']*100:.2f}   MRR@10 {s['mrr_at_10']*100:.2f}   ({elapsed:.0f}s)")

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_JSON.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)   # results contain datetimes
    print(f"wrote {RESULTS_JSON}")


if __name__ == "__main__":
    main()
