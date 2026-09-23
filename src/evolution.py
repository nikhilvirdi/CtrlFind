"""Search one version, or every version at once (the bonus goal).

All versions share a single pool of entries. An entry is a (doc id, text)
pair, so a snippet that did not change between versions is one entry that
belongs to several versions, embedded once and executed once.

Ranking across versions is where the bonus gets hard: a working version and
a buggy version of the same snippet differ by a single operator, so their
embeddings are nearly identical. Running them on the example separates
them, because the buggy one prints the wrong answer.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .execute import verify
from .versions import VersionStore, content_hash


@dataclass
class Entry:
    doc_id: str
    text: str
    versions: list[str] = field(default_factory=list)


class History:
    def __init__(self, store: VersionStore, e5):
        t0 = time.perf_counter()
        self.names = store.names()
        self.entries: list[Entry] = []
        index: dict[tuple[str, str], int] = {}
        self.rows: dict[str, np.ndarray] = {}

        for name in self.names:
            rows = []
            for d, h in store.manifest(name).items():
                key = (d, h)
                if key not in index:
                    index[key] = len(self.entries)
                    self.entries.append(Entry(d, store.text(h)))
                self.entries[index[key]].versions.append(name)
                rows.append(index[key])
            self.rows[name] = np.array(rows)

        self.emb, self.encoded = e5.encode_docs([e.text for e in self.entries])
        e5.save()
        self.order = {n: i for i, n in enumerate(self.names)}
        self.build_seconds = time.perf_counter() - t0
        self._verdicts: dict[tuple[str, str], bool] = {}

    def newest(self, i: int) -> int:
        return max(self.order[v] for v in self.entries[i].versions)

    def search(self, q_vec: np.ndarray, version: str | None, k: int):
        """Top k entries by similarity, within one version or across all."""
        rows = self.rows[version] if version else np.arange(len(self.entries))
        sims = self.emb[rows] @ q_vec
        top = np.argsort(-sims)[:k]
        return [int(rows[j]) for j in top], {int(rows[j]): float(sims[j]) for j in top}

    def verify(self, cands: list[int], example) -> dict[int, bool]:
        """Run candidates on the example, reusing verdicts for code already run."""
        ex_key = content_hash(example[0] + "\x00" + example[1])
        todo = [i for i in cands if (content_hash(self.entries[i].text), ex_key) not in self._verdicts]
        if todo:
            flags = verify([self.entries[i].text for i in todo], example)
            for i, ok in zip(todo, flags):
                self._verdicts[(content_hash(self.entries[i].text), ex_key)] = ok
        return {i: self._verdicts[(content_hash(self.entries[i].text), ex_key)] for i in cands}


def rank_similarity(cands, scores):
    return sorted(cands, key=lambda i: -scores[i])


def rank_newest(cands, scores, history: History):
    """Near-identical scores tie, and the newest version wins the tie."""
    return sorted(cands, key=lambda i: (-round(scores[i], 3), -history.newest(i)))


def rank_verified(cands, scores, passed: dict[int, bool]):
    """Versions that pass the example first, each group in similarity order.

    When execution cannot tell two versions apart, their order is exactly
    what similarity alone would give, so this never does worse than it.
    """
    return sorted(cands, key=lambda i: (not passed.get(i, False), -scores[i]))
