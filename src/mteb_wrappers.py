"""MTEB integration.

MTEB calls encode() and computes similarity itself, so a reranker cannot
live inside an encoder. It lives one level up instead: MTEB uses any model
that already satisfies SearchProtocol as-is, so a SearchEncoderWrapper
subclass passes straight through and can reorder results after the
embedding search.
"""
from __future__ import annotations

import hashlib
import json
import locale
import os
import sys
from pathlib import Path

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.models.search_wrappers import SearchEncoderWrapper
from mteb.types import PromptType

from . import execute
from .config import ADAPT_DEFAULT, ADAPT_LIMIT, DEFAULT_K, EXEC_TIMEOUT
from .encoder import E5
from .execute import VerificationOutcome, allows_trailing_padding, parse_example, verify


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

    adapt = ADAPT_DEFAULT

    def __init__(self, model, checkpoint: Path, k: int = DEFAULT_K,
                 timeout: float = EXEC_TIMEOUT, *, adapt: bool = ADAPT_DEFAULT):
        super().__init__(model)
        self.checkpoint = Path(checkpoint)
        if adapt:
            if k != DEFAULT_K:
                raise ValueError("ADAPT requires the fixed 50/200 candidate policy")
            self.checkpoint = self.checkpoint.with_name(f"{self.checkpoint.stem}.adapt{self.checkpoint.suffix}")
        self.adapt = adapt
        self.k, self.timeout = k, timeout
        self.doc_text: dict[str, str] = {}

    def index(self, corpus, **kw):
        # the parent clears task_corpus at the end of search(), so keep a copy
        self.doc_text = dict(zip(corpus["id"], corpus["text"]))
        return super().index(corpus, **kw)

    def search(self, queries, **kw):
        if self.adapt:
            return self._search_adaptive(queries, **kw)
        results = super().search(queries, **kw)
        qtext = dict(zip(queries["id"], queries["text"]))

        done: dict[str, list[str]] = {}
        if self.checkpoint.exists():
            with self.checkpoint.open(encoding="utf-8", errors="replace") as saved:
                for line in saved:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    key, ids = rec.get("verification_key"), rec.get("passer_ids")
                    if isinstance(key, str) and isinstance(ids, list) and all(isinstance(d, str) for d in ids):
                        done[key] = ids

        workers = os.cpu_count() or 2
        verifier = Path(execute.__file__)
        settings = (
            hashlib.sha256(verifier.read_bytes()).hexdigest(),
            hashlib.sha256(verifier.with_name("_windows_job.py").read_bytes()).hexdigest() if os.name == "nt" else None,
            self.timeout, workers, sys.executable, sys.version, sys.platform,
            locale.getpreferredencoding(False),
            [os.environ.get(name) for name in ("PYTHONIOENCODING", "PYTHONUTF8", "PYTHONHASHSEED")],
        )
        total = len(results)
        appended = False
        with self.checkpoint.open("a", encoding="utf-8") as out:
            for n, (qid, hits) in enumerate(results.items(), 1):
                ex = parse_example(qtext[qid])
                if ex is None or not hits:
                    continue
                padding = allows_trailing_padding(qtext[qid])
                cand = [d for d, _ in sorted(hits.items(), key=lambda x: -x[1])[: self.k]]
                codes = [self.doc_text[d] for d in cand]
                payload = json.dumps((qid, ex, padding, settings, list(zip(cand, codes))))
                key = hashlib.sha256(payload.encode("ascii")).hexdigest()
                if key not in done or not set(done[key]).issubset(cand):
                    outcomes = verify(codes, ex, self.timeout, workers=workers,
                                      allow_trailing_padding=padding, detailed=True)
                    passed_ids = [d for d, outcome in zip(cand, outcomes, strict=True)
                                  if outcome is VerificationOutcome.PASS]
                    if VerificationOutcome.INFRASTRUCTURE_ERROR not in outcomes:
                        done[key] = passed_ids
                        if not appended:
                            # Delimit a previous interrupted write before appending usable records.
                            out.write("\n")
                            appended = True
                        out.write(json.dumps({"qid": qid, "passer_ids": passed_ids,
                                              "allow_trailing_padding": padding,
                                              "verification_key": key}) + "\n")
                        out.flush()
                else:
                    passed_ids = done[key]
                if n % 100 == 0:
                    print(f"  stage 2: {n}/{total}")
                _boost(hits, set(passed_ids))
        return results

    def _search_adaptive(self, queries, *, top_k, **kw):
        qtext = dict(zip(queries["id"], queries["text"], strict=True))
        if top_k <= 0 or not self.doc_text:
            return {qid: {} for qid in qtext}
        # MTEB may discard exact ties at a truncated search boundary.
        hits_by_query = super().search(queries, top_k=len(self.doc_text), **kw)
        workers = os.cpu_count() or 2
        verifier = Path(execute.__file__)
        settings = (
            hashlib.sha256(verifier.read_bytes()).hexdigest(),
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            hashlib.sha256(verifier.with_name("_windows_job.py").read_bytes()).hexdigest() if os.name == "nt" else None,
            self.timeout, workers, sys.executable, sys.version, sys.platform,
            locale.getpreferredencoding(False),
            sys.flags.utf8_mode, sorted(os.environ.items()),
        )
        done = {}
        if self.checkpoint.exists():
            with self.checkpoint.open(encoding="utf-8", errors="replace") as saved:
                for line in saved:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(rec, dict) and rec.get("policy") == "adapt-v1" and isinstance(rec.get("key"), str):
                        done[rec["key"]] = rec

        results = {}
        for qid, hits in hits_by_query.items():
            ordered = sorted(hits, key=lambda did: (-hits[did], did))
            ex = parse_example(qtext[qid])
            if ex is None or not ordered:
                results[qid] = _rank_scores(ordered[:top_k])
                continue
            padding = allows_trailing_padding(qtext[qid])
            candidates = ordered[:ADAPT_LIMIT]
            prefix = candidates[:DEFAULT_K]
            payload = ("adapt-v1", qid, ex, padding, settings, [(did, self.doc_text[did]) for did in candidates])
            key = hashlib.sha256(json.dumps(payload).encode("ascii")).hexdigest()
            record = done.get(key, {})
            attempted, passers = record.get("attempted_ids"), record.get("passer_ids")
            if not _valid_adapt_record(attempted, passers, prefix, candidates):
                attempted = prefix
                suffix = candidates[len(prefix):]
                try:
                    outcomes = verify([self.doc_text[did] for did in prefix], ex, self.timeout, workers=workers,
                                      allow_trailing_padding=padding, detailed=True)
                    if VerificationOutcome.PASS not in outcomes and suffix:
                        extra = verify([self.doc_text[did] for did in suffix], ex, self.timeout, workers=workers,
                                       allow_trailing_padding=padding, detailed=True)
                        outcomes = outcomes + extra
                        attempted = candidates
                except execute.VerificationInfrastructureError as exc:
                    raise execute.VerificationInfrastructureError(f"ADAPT verification failed for query {qid}") from exc
                passers = [did for did, outcome in zip(attempted, outcomes, strict=True)
                           if outcome is VerificationOutcome.PASS]
                if VerificationOutcome.INFRASTRUCTURE_ERROR not in outcomes:
                    record = {"policy": "adapt-v1", "key": key, "attempted_ids": attempted, "passer_ids": passers}
                    with self.checkpoint.open("a", encoding="utf-8") as out:
                        out.write("\n" + json.dumps(record) + "\n")
                    done[key] = record
            passing = set(passers)
            ranked = passers + [did for did in ordered if did not in passing]
            results[qid] = _rank_scores(ranked[:top_k])
        return results


def _rank_scores(ordered):
    # MTEB breaks numeric ties by descending ID. Encode the order decided above.
    return {did: float(-position) for position, did in enumerate(ordered)}


def _valid_adapt_record(attempted, passers, prefix, candidates):
    if attempted not in (prefix, candidates) or not isinstance(passers, list):
        return False
    if not all(isinstance(did, str) for did in passers):
        return False
    passing = set(passers)
    if passers != [did for did in attempted if did in passing]:
        return False
    if prefix == candidates:
        return True
    if attempted == prefix:
        return bool(passers)
    return passing.isdisjoint(prefix)
