"""MTEB integration.

MTEB calls encode() and computes similarity itself, so a reranker cannot
live inside an encoder. It lives one level up instead: MTEB uses any model
that already satisfies SearchProtocol as-is, so a SearchEncoderWrapper
subclass passes straight through and can reorder results after the
embedding search.
"""
from __future__ import annotations

import json
from pathlib import Path

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.models.search_wrappers import SearchEncoderWrapper
from mteb.types import PromptType

from .config import DEFAULT_K, EXEC_TIMEOUT
from .encoder import E5
from .execute import parse_example, verify


def make_meta(name: str) -> ModelMeta:
    # loader and memory_usage_mb are required in mteb 2.21. A missing meta
    # lets evaluation run and then crashes in the result cache.
    return ModelMeta(
        name=name, revision="1", release_date="2026-09-24",
        languages=["eng-Latn"], loader=None, memory_usage_mb=418,
        n_parameters=110_000_000, max_tokens=512, embed_dim=768,
        license="mit", open_weights=True, public_training_code=None,
        public_training_data=None, framework=["Sentence Transformers"],
        similarity_fn_name="cosine", use_instructions=True,
        training_datasets=None,
    )


class E5Encoder(AbsEncoder):
    def __init__(self, e5: E5, name: str):
        self.e5 = e5
        self.model = e5.model
        self.mteb_model_meta = make_meta(name)

    def encode(self, inputs, *, task_metadata, hf_split, hf_subset,
               prompt_type=None, **kwargs):
        texts = [t for batch in inputs for t in batch["text"]]
        if prompt_type == PromptType.query:
            emb, n = self.e5.encode_queries(texts)
        else:
            emb, n = self.e5.encode_docs(texts)
        self.e5.save()
        print(f"  {len(texts)} texts, {n} encoded, {len(texts) - n} from cache")
        return emb


def _boost(hits: dict, passed_ids) -> None:
    """Move passing docs above everything else, keeping relative order."""
    if not passed_ids or not hits:
        return
    boost = max(hits.values()) + 1.0
    for d in passed_ids:
        if d in hits:
            hits[d] = boost + hits[d]


class CachedExecReranker(SearchEncoderWrapper):
    """Applies stage 2 verdicts recorded by an earlier full run."""

    def __init__(self, model, passed_by_qid: dict[str, set[str]]):
        super().__init__(model)
        self.passed = passed_by_qid

    def search(self, queries, **kw):
        results = super().search(queries, **kw)
        for qid, hits in results.items():
            _boost(hits, self.passed.get(qid))
        return results


class LiveExecReranker(SearchEncoderWrapper):
    """Runs stage 2 for real. Resumable through a JSONL checkpoint."""

    def __init__(self, model, checkpoint: Path, k: int = DEFAULT_K,
                 timeout: float = EXEC_TIMEOUT):
        super().__init__(model)
        self.checkpoint = Path(checkpoint)
        self.k, self.timeout = k, timeout
        self.doc_text: dict[str, str] = {}

    def index(self, corpus, **kw):
        # the parent clears task_corpus at the end of search(), so keep a copy
        self.doc_text = dict(zip(corpus["id"], corpus["text"]))
        return super().index(corpus, **kw)

    def search(self, queries, **kw):
        results = super().search(queries, **kw)
        qtext = dict(zip(queries["id"], queries["text"]))

        done: dict[str, list[str]] = {}
        if self.checkpoint.exists():
            for line in self.checkpoint.open(encoding="utf-8"):
                rec = json.loads(line)
                done[rec["qid"]] = rec["passer_ids"]

        total = len(results)
        with self.checkpoint.open("a", encoding="utf-8") as out:
            for n, (qid, hits) in enumerate(results.items(), 1):
                if qid not in done:
                    passed_ids: list[str] = []
                    ex = parse_example(qtext[qid])
                    if ex:
                        cand = [d for d, _ in sorted(hits.items(), key=lambda x: -x[1])[: self.k]]
                        flags = verify([self.doc_text[d] for d in cand], ex, self.timeout)
                        passed_ids = [d for d, ok in zip(cand, flags) if ok]
                    done[qid] = passed_ids
                    out.write(json.dumps({"qid": qid, "passer_ids": passed_ids}) + "\n")
                    out.flush()
                if n % 100 == 0:
                    print(f"  stage 2: {n}/{total}")
                _boost(hits, set(done[qid]))
        return results
