"""Stage 1: E5-base-v2 embeddings, backed by the content-addressed cache."""
from __future__ import annotations

import numpy as np

from .cache import EmbeddingCache
from .config import (DOC_PREFIX, EMB_CACHE, MAX_SEQ_LEN, MODEL_NAME,
                     QUERY_PREFIX)


def pick_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class E5:
    """Thin wrapper: prefixes, normalisation, and cache lookups."""

    def __init__(self, device: str | None = None, cache_path=EMB_CACHE):
        from sentence_transformers import SentenceTransformer

        self.device = device or pick_device()
        self.model = SentenceTransformer(MODEL_NAME, device=self.device)
        self.model.max_seq_length = MAX_SEQ_LEN
        self.cache = EmbeddingCache(cache_path, MODEL_NAME)
        self.batch_size = 64 if self.device == "cuda" else 16

    def _encode_raw(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 64,
        )

    def encode_docs(self, texts) -> tuple[np.ndarray, int]:
        return self.cache.get_many([DOC_PREFIX + t for t in texts], self._encode_raw)

    def encode_queries(self, texts) -> tuple[np.ndarray, int]:
        return self.cache.get_many([QUERY_PREFIX + t for t in texts], self._encode_raw)

    def save(self) -> None:
        self.cache.save()
