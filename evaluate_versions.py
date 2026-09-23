"""Measure retrieval across versions (P1) and across all versions at once (bonus).

  python evaluate_versions.py            run both, write artifacts/versions_report.json
  python evaluate_versions.py --n 100    fewer queries for the bonus test

Needs: python versions.py build
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from src.config import DEFAULT_K, VERSIONS_DIR, VERSIONS_REPORT
from src.data import load_apps
from src.encoder import E5
from src.evolution import History, rank_newest, rank_similarity, rank_verified
from src.execute import parse_example, verify
from src.versions import VersionStore


def ndcg10(ranked: list[int], relevant: set[int]) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1 / np.log2(r + 2) for r, i in enumerate(ranked[:10]) if i in relevant)
    idcg = sum(1 / np.log2(r + 2) for r in range(min(len(relevant), 10)))
    return dcg / idcg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    args = ap.parse_args()

    store = VersionStore(VERSIONS_DIR)
    if not store.exists():
        raise SystemExit("No versions yet. Run: python versions.py build")
    bugs = json.loads((VERSIONS_DIR / "bugs.json").read_text())
    tracked = json.loads((VERSIONS_DIR / "tracked.json").read_text())

    apps = load_apps()
    e5 = E5()
    hist = History(store, e5)
    print(f"{len(hist.entries)} entries across {len(hist.names)} versions, "
          f"{hist.encoded} newly encoded, {hist.build_seconds:.1f}s")

    q_text = dict(zip(apps.query_ids, apps.query_texts))
    q_emb, _ = e5.encode_queries(apps.query_texts)
    e5.save()
    q_vec = dict(zip(apps.query_ids, q_emb))

    def buggy(i: int) -> bool:
        e = hist.entries[i]
        return bugs[e.versions[0]][e.doc_id]

    report: dict = {"entries": len(hist.entries), "p1": {}, "bonus": {}}

    # P1: retrieval on each version, stage 1 only ------------------------
    print("\nP1, retrieval on each version (stage 1, all test queries)")
    for name in hist.names:
        rows = hist.rows[name]
        pos = {hist.entries[i].doc_id: j for j, i in enumerate(rows)}
        sub = hist.emb[rows]
        t0 = time.perf_counter()
        scores = []
        for q in apps.query_ids:
            sims = sub @ q_vec[q]
            g = pos[apps.gold[q]]
            rank = int((sims > sims[g]).sum()) + 1
            scores.append(1 / np.log2(rank + 1) if rank <= 10 else 0.0)
        val = float(np.mean(scores) * 100)
        report["p1"][name] = {"ndcg_at_10": round(val, 2),
                              "search_seconds": round(time.perf_counter() - t0, 2)}
        print(f"   {name}: NDCG@10 {val:.2f}")

    # Bonus: all versions at once ----------------------------------------
    print("\nBonus, searching all versions at once")
    usable = []
    for q in tracked:
        ex = parse_example(q_text[q])
        if not ex:
            continue
        # keep problems whose original answer passes its own example,
        # otherwise working and buggy versions cannot be told apart by anyone
        if verify([store.text(store.manifest("v1")[apps.gold[q]])], ex)[0]:
            usable.append((q, ex))
        if len(usable) >= args.n:
            break
    print(f"   {len(usable)} problems with a usable example")

    methods = {"similarity": [], "similarity + newest": [], "similarity + execution": []}
    ndcg = {m: [] for m in methods}
    detected, total_bug = 0, 0
    t0 = time.perf_counter()
    for n, (q, ex) in enumerate(usable, 1):
        cands, scores = hist.search(q_vec[q], None, args.k)
        passed = hist.verify(cands, ex)
        fam = {i for i in cands if hist.entries[i].doc_id == apps.gold[q]}
        good = {i for i in fam if not buggy(i)}
        for i in fam - good:
            total_bug += 1
            detected += not passed[i]
        rankings = {
            "similarity": rank_similarity(cands, scores),
            "similarity + newest": rank_newest(cands, scores, hist),
            "similarity + execution": rank_verified(cands, scores, passed),
        }
        for m, ranked in rankings.items():
            ndcg[m].append(ndcg10(ranked, good))
            first = next((i for i in ranked if i in fam), None)
            if first is not None and fam - good:
                methods[m].append(first in good)
        if n % 25 == 0:
            print(f"   {n}/{len(usable)}")

    print(f"\n   {'method':<26}{'working version first':>24}{'NDCG@10':>10}")
    for m in methods:
        acc = float(np.mean(methods[m]) * 100) if methods[m] else 0.0
        nd = float(np.mean(ndcg[m]) * 100) if ndcg[m] else 0.0
        report["bonus"][m] = {"working_version_first_pct": round(acc, 1),
                              "ndcg_at_10": round(nd, 2)}
        print(f"   {m:<26}{acc:>23.1f}%{nd:>10.2f}")
    contested = len(methods["similarity"])
    report["bonus"]["problems"] = len(usable)
    report["bonus"]["contested_problems"] = contested
    report["bonus"]["buggy_versions_caught_by_example"] = f"{detected}/{total_bug}"
    report["bonus"]["seconds"] = round(time.perf_counter() - t0, 1)
    print(f"\n   contested problems (a buggy version was in the top {args.k}): {contested}")
    print(f"   buggy versions that failed the example: {detected}/{total_bug}")

    VERSIONS_REPORT.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {VERSIONS_REPORT}")


if __name__ == "__main__":
    main()
