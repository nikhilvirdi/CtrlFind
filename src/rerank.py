"""Combining stage 1 order with stage 2 verdicts."""
from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def rerank(candidates: Sequence[T], passed: set[T]) -> list[T]:
    """Passing candidates first, each group keeping its embedding order."""
    if not passed:
        return list(candidates)
    top = [c for c in candidates if c in passed]
    rest = [c for c in candidates if c not in passed]
    return top + rest
