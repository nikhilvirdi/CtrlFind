"""Produce the MTEB results file for AppsRetrieval.

  python evaluate.py --mode cached     replay historical verdicts without exit-code provenance
  python evaluate.py --mode full       run stage 2 live (hours on CPU, resumable)
  python evaluate.py --mode baseline   stage 1 only

Writes a new report in artifacts/evaluation_runs/ or at --output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from src.config import ADAPT_DEFAULT, ARTIFACTS, DEFAULT_K, DOC_IDS, EXEC_RESULTS
from src.evaluation_output import evaluation_encoder, report_path, write_report


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
    ap.add_argument("--mode", choices=["cached", "full", "baseline"], required=True)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--adapt", action="store_true", default=ADAPT_DEFAULT,
                    help="Verify ranks 51–200 only when none of the first 50 passes")
    ap.add_argument("--output", type=Path, help="Path to a new JSON report")
    args = ap.parse_args()
    if args.adapt and (args.mode != "full" or args.k != DEFAULT_K):
        ap.error("--adapt requires --mode full and --k 50")
    policy = "full_adapt" if args.adapt else args.mode
    output = report_path(ap, args.output, f"appsretrieval_{policy}.json")

    import mteb
    from src.mteb_wrappers import (CachedExecReranker, E5Encoder,
                                   LiveExecReranker)

    e5 = evaluation_encoder()
    print(f"device: {e5.device}, cached embeddings: {len(e5.cache)}")

    if args.mode == "baseline":
        model = E5Encoder(e5, "prism/e5-base-v2-baseline")
    elif args.mode == "cached":
        verdicts = load_cached_verdicts()
        print(f"historical replay without exit-code provenance: {len(verdicts)} queries with passing snippets")
        model = CachedExecReranker(E5Encoder(e5, "prism/e5-base-v2-exec-rerank"), verdicts)
    else:
        suffix = "-adapt" if args.adapt else ""
        model = LiveExecReranker(E5Encoder(e5, f"prism/e5-base-v2-exec-rerank-live{suffix}"),
                                 ARTIFACTS / "exec_results_live.jsonl", k=args.k, adapt=args.adapt)

    t0 = time.time()
    # Full mode resumes candidate checks from its own checkpoint, not an old MTEB score.
    cache_options = {"cache": None} if args.mode == "full" else {}
    result = mteb.evaluate(model, [mteb.get_task("AppsRetrieval")],
                           encode_kwargs={"batch_size": e5.batch_size}, **cache_options)
    elapsed = time.time() - t0

    out = list(result.task_results)[0].to_dict()
    s = out["scores"]["test"][0]
    print(f"\nNDCG@10 {s['ndcg_at_10']*100:.2f}   MRR@10 {s['mrr_at_10']*100:.2f}   ({elapsed:.0f}s)")

    write_report(output, out)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
