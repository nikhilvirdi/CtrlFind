"""Content-addressed embedding cache.

Every text is keyed by a hash of (model name, prefixed text). Encoding a
corpus only runs the model on texts whose hash is not already stored, so
an edited snippet costs one re-encode and an unchanged corpus costs none.
This is what makes re-indexing a new code version cheap (goal P1).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


def text_key(model_name: str, text: str) -> str:
    return hashlib.sha1(f"{model_name}\x00{text}".encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, path: Path, model_name: str):
        self.path = Path(path)
        self.model_name = model_name
        self._index: dict[str, int] = {}
        self._vecs: list[np.ndarray] = []
        self.dirty = False
        if self.path.exists():
            z = np.load(self.path, allow_pickle=False)
            keys, vecs = z["keys"], z["vecs"]
            self._vecs = [vecs]
            self._index = {k: i for i, k in enumerate(keys.tolist())}

    def __len__(self) -> int:
        return len(self._index)

    def _matrix(self) -> np.ndarray:
        if len(self._vecs) > 1:
            self._vecs = [np.concatenate(self._vecs, axis=0)]
        return self._vecs[0] if self._vecs else np.zeros((0, 0), dtype=np.float32)

    def get_many(
        self,
        texts: Sequence[str],
        encode_fn: Callable[[list[str]], np.ndarray],
    ) -> tuple[np.ndarray, int]:
        """Return embeddings for texts, encoding only the misses.

        Returns (embeddings, number_of_newly_encoded_texts).
        """
        keys = [text_key(self.model_name, t) for t in texts]
        missing, seen = [], set()
        for k, t in zip(keys, texts):
            if k not in self._index and k not in seen:
                missing.append((k, t))
                seen.add(k)

        if missing:
            new = np.asarray(encode_fn([t for _, t in missing]), dtype=np.float32)
            start = len(self._index)
            for offset, (k, _) in enumerate(missing):
                self._index[k] = start + offset
            self._vecs.append(new)
            self.dirty = True

        mat = self._matrix()
        return mat[[self._index[k] for k in keys]], len(missing)

    def save(self) -> None:
        if not self.dirty:
            return
        mat = self._matrix()
        keys = np.empty(len(self._index), dtype=object)
        for k, i in self._index.items():
            keys[i] = k
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")
        np.savez(tmp, keys=keys.astype(str), vecs=mat)
        tmp.replace(self.path)
        self.dirty = False
