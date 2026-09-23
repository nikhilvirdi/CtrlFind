"""Loading CoIR APPS from HuggingFace."""
from __future__ import annotations

from dataclasses import dataclass

from .config import DATASET


@dataclass
class Apps:
    doc_ids: list[str]
    doc_texts: list[str]
    query_ids: list[str]          # test split only
    query_texts: list[str]
    gold: dict[str, str]          # query id -> relevant doc id


def load_apps() -> Apps:
    from datasets import load_dataset

    corpus = load_dataset(DATASET, "corpus", split="corpus")
    queries = load_dataset(DATASET, "queries", split="queries")
    qrels = load_dataset(DATASET, "default", split="test")

    test_q = queries.filter(lambda r: r["partition"] == "test")
    gold = {r["query-id"]: r["corpus-id"] for r in qrels}

    return Apps(
        doc_ids=list(corpus["_id"]),
        doc_texts=list(corpus["text"]),
        query_ids=list(test_q["_id"]),
        query_texts=list(test_q["text"]),
        gold=gold,
    )
