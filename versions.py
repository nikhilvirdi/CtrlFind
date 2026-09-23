"""Build and inspect the version history (goal P1).

  python versions.py build          generate v1, v2, v3 and index each one
  python versions.py list           show versions
  python versions.py diff v1 v2     compare two versions
"""
from __future__ import annotations

import argparse
import json
import random
import time

from src.config import VERSIONS_DIR
from src.versions import VersionStore


def build(tracked_n: int, seed: int) -> None:
    from src.data import load_apps
    from src.encoder import E5
    from src.execute import parse_example, verify
    from src.history import make_history

    if VERSIONS_DIR.exists() and any(VERSIONS_DIR.iterdir()):
        raise SystemExit(f"{VERSIONS_DIR} already exists. Delete it to rebuild the history.")

    apps = load_apps()
    rng = random.Random(seed)
    tracked_q = rng.sample(apps.query_ids, tracked_n)
    tracked_docs = {apps.gold[q] for q in tracked_q}

    base = dict(zip(apps.doc_ids, apps.doc_texts))

    # example for each tracked snippet, taken from the problem it answers
    q_text = dict(zip(apps.query_ids, apps.query_texts))
    example = {apps.gold[q]: ex for q in tracked_q if (ex := parse_example(q_text[q]))}

    def observable(doc_id: str, code: str) -> bool:
        ex = example.get(doc_id)
        return ex is not None and not verify([code], ex)[0]

    print(f"generating history, {len(tracked_docs)} tracked snippets")
    versions, bugs = make_history(base, tracked_docs, seed=seed, observable=observable)

    store = VersionStore(VERSIONS_DIR)
    print("committing versions")
    for name in ["v1", "v2", "v3"]:
        print("  ", store.commit(name, versions[name]))
    (VERSIONS_DIR / "bugs.json").write_text(json.dumps(bugs))
    (VERSIONS_DIR / "tracked.json").write_text(json.dumps(tracked_q))

    print("\nindexing each version (only changed snippets are encoded)")
    e5 = E5()
    for name in store.names():
        docs = store.checkout(name)
        t0 = time.perf_counter()
        _, n = e5.encode_docs(list(docs.values()))
        e5.save()
        print(f"   {name}: {n} encoded, {len(docs) - n} reused, {time.perf_counter() - t0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--tracked", type=int, default=300,
                   help="test queries whose answers get a distinct v2 and v3")
    b.add_argument("--seed", type=int, default=0)
    sub.add_parser("list")
    d = sub.add_parser("diff")
    d.add_argument("a"); d.add_argument("b")
    args = ap.parse_args()

    if args.cmd == "build":
        build(args.tracked, args.seed)
        return
    store = VersionStore(VERSIONS_DIR)
    if not store.exists():
        raise SystemExit("No versions yet. Run: python versions.py build")
    if args.cmd == "list":
        for n in store.names():
            print(n, len(store.manifest(n)), "snippets")
    else:
        print(store.diff(args.a, args.b))


if __name__ == "__main__":
    main()
