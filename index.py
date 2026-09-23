"""Build or refresh the document index (goal P1: retrieval across versions).

Embeddings are cached by content hash, so a new version of the codebase
only re-encodes the snippets that actually changed.

  python index.py                        build or refresh the index
  python index.py --simulate-edits 100   edit 100 random snippets and time the rebuild
"""
from __future__ import annotations

import argparse
import random
import time

from src.data import load_apps
from src.encoder import E5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate-edits", type=int, default=0,
                    help="modify N random snippets to measure an incremental rebuild")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    apps = load_apps()
    e5 = E5()
    print(f"corpus: {len(apps.doc_texts)} snippets, device: {e5.device}")

    t0 = time.perf_counter()
    _, n = e5.encode_docs(apps.doc_texts)
    e5.save()
    print(f"index ready: {n} encoded, {len(apps.doc_texts) - n} reused, "
          f"{time.perf_counter() - t0:.1f}s")

    if args.simulate_edits:
        rng = random.Random(args.seed)
        texts = list(apps.doc_texts)
        for i in rng.sample(range(len(texts)), args.simulate_edits):
            texts[i] = texts[i] + f"\n# revised {i}\n"
        t0 = time.perf_counter()
        _, n = e5.encode_docs(texts)
        print(f"new version with {args.simulate_edits} edited snippets: "
              f"{n} re-encoded, {len(texts) - n} reused, "
              f"{time.perf_counter() - t0:.2f}s")
        # the simulated version is not saved, so the stored index stays clean


if __name__ == "__main__":
    main()
