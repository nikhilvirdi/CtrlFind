"""Audit the training census and two saved rank ledgers for the fixed training sample."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import random
from urllib.parse import urlsplit

import numpy as np
import pyarrow.parquet as pq

from verification_research import SNAPSHOT, load_train
from src.config import DATASET, DEFAULT_K, MAX_SEQ_LEN, MODEL_NAME, QUERY_PREFIX
from src.execute import parse_example

HERE = Path(__file__).resolve().parent
SEED = 20261125
MODEL_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
SAMPLE_SIZE = 500
PERMUTATION_SEED = SEED + 1
PERMUTATIONS = 20000
OLD_RANKS = HERE / "retrieval_diagnostic_train.json"
TRAIN_RANKS = HERE / "query_example_baseline_train.json"
HIT_KEYS = ("hit_prior_full_corpus", "hit_train_corpus")


def load_rows():
    from transformers import AutoTokenizer

    queries, docs, gold = load_train(SNAPSHOT)
    qids = random.Random(SEED).sample(sorted(queries), SAMPLE_SIZE)
    previous_bytes, train_bytes = OLD_RANKS.read_bytes(), TRAIN_RANKS.read_bytes()
    previous, train = json.loads(previous_bytes), json.loads(train_bytes)
    assert previous["query_split"] == "train"
    assert previous["dataset_revision"] == train["train_revision"] == SNAPSHOT.name
    assert qids == previous["sampling"]["query_ids"] == train["populations"]["uniform"]
    assert train["k"] == DEFAULT_K and train["corpus_count"] == len(docs)
    assert train["model_revision"] == previous["model"]["revision"] == MODEL_REVISION
    assert previous["model"]["max_seq_length"] == MAX_SEQ_LEN
    corpus_digest = hashlib.sha256("\n".join(sorted(docs)).encode()).hexdigest()
    assert corpus_digest == train["corpus_sha256"]

    (path,) = SNAPSHOT.glob("queries/*.parquet")
    metadata = pq.read_table(path, columns=["_id", "meta_information"],
                             filters=[("partition", "=", "train")]).to_pylist()
    metadata = {item["_id"]: item["meta_information"] for item in metadata}
    assert metadata.keys() == queries.keys()
    model_path = Path.home() / ".cache/huggingface/hub/models--intfloat--e5-base-v2/snapshots" / MODEL_REVISION
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    ids, sampled = sorted(queries), set(qids)
    rows = []
    for start in range(0, len(ids), 64):
        batch = ids[start:start + 64]
        formatted = [QUERY_PREFIX + queries[qid] for qid in batch]
        token_ids = tokenizer(formatted, add_special_tokens=True, truncation=False)["input_ids"]
        for qid, tokens in zip(batch, token_ids, strict=True):
            text, info = queries[qid], metadata[qid]
            parsed = parse_example(text) is not None
            row = {
                "query_id": qid, "source": urlsplit(info["url"]).hostname.removeprefix("www."),
                "sampled": qid in sampled, "parsed": parsed, "characters": len(text),
                "formatted_tokens": len(tokens), "truncated": len(tokens) > MAX_SEQ_LEN,
                "starter_code": bool((info["starter_code"] or "").strip()),
            }
            if qid in sampled:
                old, new = previous["rows"][qid], train["rows"][qid]
                assert old["gold_id"] == gold[qid] and old["example_found"] == parsed
                assert old["query_tokens"][1:] == [len(tokens), min(len(tokens), MAX_SEQ_LEN)]
                prior_rank, train_rank = old["stage1_gold_rank"], new["rank"]
                assert len(new["top"]) == len(set(new["top"])) == DEFAULT_K
                assert set(new["top"]) <= docs.keys() and 1 <= train_rank <= len(docs)
                assert (new["top"].index(gold[qid]) + 1 if gold[qid] in new["top"] else None) == (
                    train_rank if train_rank <= DEFAULT_K else None)
                row.update(prior_full_corpus_rank=prior_rank, train_corpus_rank=train_rank,
                           hit_prior_full_corpus=prior_rank is not None and prior_rank <= DEFAULT_K,
                           hit_train_corpus=train_rank <= DEFAULT_K)
            rows.append(row)
    provenance = {
        "prior_full_corpus_training_rank_ledger": {"file": OLD_RANKS.name, "sha256": hashlib.sha256(previous_bytes).hexdigest(),
                                                   "corpus_size": previous["corpus"]["count"]},
        "training_only_rank_ledger": {"file": TRAIN_RANKS.name, "sha256": hashlib.sha256(train_bytes).hexdigest(),
                                      "corpus_size": len(docs), "ordered_document_ids_sha256": corpus_digest},
        "model": MODEL_NAME, "model_revision": MODEL_REVISION, "query_prefix": QUERY_PREFIX,
        "max_seq_len": MAX_SEQ_LEN, "tokenizer": type(tokenizer).__name__,
        "runtime": {"python": platform.python_version(), "packages": {
            name: version(name) for name in ("numpy", "pyarrow", "transformers", "tokenizers")}},
    }
    return rows, qids, provenance


def rate(count, n, interval=False):
    p = count / n if n else None
    result = {"count": count, "denominator": n, "percent": 100 * p if n else None}
    if interval and n:
        z = 1.959963984540054
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        result["wilson_95_percent"] = [100 * max(0., center - half), 100 * min(1., center + half)]
    return result


def profile(rows, sampled=False):
    result = {"n": len(rows), **{flag: rate(sum(row[flag] for row in rows), len(rows), sampled)
                                 for flag in ("parsed", "truncated", "starter_code")}}
    for key in ("characters", "formatted_tokens"):
        values = [row[key] for row in rows]
        result[key] = {"mean": float(np.mean(values)), "p25_median_p75_p95": np.quantile(values, [.25, .5, .75, .95]).tolist()}
    return result


def summarize_hits(rows, key, total_misses):
    hits = sum(row[key] for row in rows)
    misses = len(rows) - hits
    return {"recall_at_50": rate(hits, len(rows), interval=True), "misses": rate(misses, len(rows)),
            "share_of_all_misses": rate(misses, total_misses)}


def token_band(row):
    return ("<=256", "257-512", ">512")[int(row["formatted_tokens"] > 256) + int(row["truncated"])]


def stratified_gap(rows, flag, strata, hit):
    """Compare the observed gap to the exact null conditional on group margins."""
    groups = defaultdict(list)
    for row in rows:
        groups[strata(row)].append(row)
    distribution = np.array([1.0])
    exposed = hits = observed = shared_groups = shared_n = 0
    expected = 0.0
    for members in groups.values():
        n = len(members)
        a = sum(row[flag] for row in members)
        h = sum(row[hit] for row in members)
        exposed += a
        hits += h
        observed += sum(row[flag] and row[hit] for row in members)
        expected += a * h / n
        if 0 < a < n:
            shared_groups += 1
            shared_n += n
        denominator = math.comb(n, a)
        probabilities = [math.comb(h, x) * math.comb(n - h, a - x) / denominator for x in range(a + 1)]
        distribution = np.convolve(distribution, probabilities)
    assert np.isclose(distribution.sum(), 1.)
    unexposed = len(rows) - exposed
    observed_gap = 100 * (observed / exposed - (hits - observed) / unexposed)
    expected_gap = 100 * (expected / exposed - (hits - expected) / unexposed)
    tail = np.abs(np.arange(len(distribution)) - expected) >= abs(observed - expected) - 1e-12
    return {"observed_gap_pp": observed_gap, "composition_expected_gap_pp": expected_gap,
            "residual_gap_pp": observed_gap - expected_gap, "expected_flagged_hits": expected,
            "flagged_hits": rate(observed, exposed, interval=True),
            "other_hits": rate(hits - observed, unexposed, interval=True),
            "overlap_strata": shared_groups, "total_strata": len(groups), "queries_in_overlap_strata": shared_n,
            "exact_conditional_p": min(1., float(distribution[tail].sum()))}


def ks_to_census(sample, population):
    unique, counts = np.unique(sample, return_counts=True)
    right = np.cumsum(counts)
    differences = [np.max(np.abs(cdf / len(sample) - np.searchsorted(population, unique, side=side) / len(population)))
                   for side, cdf in (("left", right - counts), ("right", right))]
    return max(differences) / (1 - len(sample) / len(population))


def representativeness(rows):
    names = ("source_mix_pearson", "parser_count_deviation", "truncation_count_deviation",
             "character_length_ks", "formatted_token_length_ks")
    _, sources = np.unique([row["source"] for row in rows], return_inverse=True)
    sample = np.flatnonzero([row["sampled"] for row in rows])
    n = len(sample)
    expected_sources = np.bincount(sources) * n / len(rows)
    binary = np.array([[row["parsed"], row["truncated"]] for row in rows])
    expected_binary = binary.sum(axis=0) * n / len(rows)
    lengths = [np.array([row[key] for row in rows]) for key in ("characters", "formatted_tokens")]
    sorted_lengths = [np.sort(values) for values in lengths]

    def statistics(indices):
        source_deviation = np.bincount(sources[indices], minlength=len(expected_sources)) - expected_sources
        return np.array([(source_deviation ** 2 / expected_sources).sum(),
                         *np.abs(binary[indices].sum(axis=0) - expected_binary),
                         *(ks_to_census(values[indices], population) for values, population in zip(lengths, sorted_lengths))])

    observed = statistics(sample)
    rng = np.random.default_rng(PERMUTATION_SEED)
    extreme = np.zeros(len(names), dtype=int)
    for _ in range(PERMUTATIONS):
        extreme += statistics(rng.choice(len(rows), size=n, replace=False)) >= observed - 1e-12
    p = (extreme + 1) / (PERMUTATIONS + 1)
    order = np.argsort(p)
    holm = np.empty_like(p)
    holm[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return {name: {"statistic": float(stat), "permutation_p": float(raw), "holm_p": float(adjusted)}
            for name, stat, raw, adjusted in zip(names, observed, p, holm, strict=True)}


def summarize(rows, qids, provenance):
    sample = [row for row in rows if row["sampled"]]
    complement = [row for row in rows if not row["sampled"]]
    sample_groups, census_groups = defaultdict(list), defaultdict(list)
    for row in rows:
        census_groups[row["source"]].append(row)
        if row["sampled"]:
            sample_groups[row["source"]].append(row)

    summary = {"dataset": DATASET, "revision": SNAPSHOT.name, "split": "train", "provenance": provenance,
               "sampling": {"seed": SEED, "query_ids": qids,
                            "ordered_ids_sha256": hashlib.sha256("\n".join(qids).encode()).hexdigest()},
               "profiles": {"sample": profile(sample, True), "complement": profile(complement), "census": profile(rows)},
               "representativeness": {"sample_size": len(sample), "census_size": len(rows),
                                      "permutations": PERMUTATIONS, "seed": PERMUTATION_SEED,
                                      "tests": representativeness(rows)},
               "sources": {}, "retrieval": {}}
    for source in sorted(census_groups):
        group, census = sample_groups[source], census_groups[source]
        summary["sources"][source] = {"sample_share": rate(len(group), len(sample)),
                                      "census_share": rate(len(census), len(rows)),
                                      "sample": profile(group, True), "census": profile(census)}
    for hit in HIT_KEYS:
        misses = sum(not row[hit] for row in sample)
        breakdown = {"all": summarize_hits(sample, hit, misses), "sources": {}, "groups": {}, "associations": {}}
        breakdown["miss_overlap"] = {
            "parsed_and_truncated": rate(sum(row["parsed"] and row["truncated"] and not row[hit] for row in sample), misses),
            "neither_parsed_nor_truncated": rate(sum(not (row["parsed"] or row["truncated"]) and not row[hit] for row in sample), misses),
        }
        for source, group in sorted(sample_groups.items()):
            cells = []
            for parsed in (False, True):
                for truncated in (False, True):
                    subgroup = [row for row in group if row["parsed"] == parsed and row["truncated"] == truncated]
                    cells.append({"parsed": parsed, "truncated": truncated, **summarize_hits(subgroup, hit, misses)})
            breakdown["sources"][source] = {
                **summarize_hits(group, hit, misses), "cells": cells,
                "groups": {flag: {str(value).lower(): summarize_hits([row for row in group if row[flag] == value], hit, misses)
                                  for value in (False, True)} for flag in ("parsed", "truncated")},
            }
        for flag in ("parsed", "truncated", "starter_code"):
            breakdown["groups"][flag] = {}
            for value in (False, True):
                group = [row for row in sample if row[flag] == value]
                breakdown["groups"][flag][str(value).lower()] = {**profile(group, True), **summarize_hits(group, hit, misses)}
        breakdown["groups"]["token_band"] = {
            band: summarize_hits([row for row in sample if token_band(row) == band], hit, misses)
            for band in ("<=256", "257-512", ">512")
        }
        for flag, options in {
            "parsed": {"source": lambda r: r["source"], "truncation": lambda r: r["truncated"],
                       "starter_code": lambda r: r["starter_code"],
                       "source_and_truncation": lambda r: (r["source"], r["truncated"]),
                       "source_and_token_band": lambda r: (r["source"], token_band(r))},
            "truncated": {"source": lambda r: r["source"], "source_and_parser": lambda r: (r["source"], r["parsed"])},
        }.items():
            breakdown["associations"][flag] = {name: stratified_gap(sample, flag, strata, hit) for name, strata in options.items()}
        summary["retrieval"][hit] = breakdown

    before, after = HIT_KEYS
    transitions = Counter((row[before], row[after]) for row in sample)
    summary["paired_protocol_transition"] = {"both_hit": transitions[True, True], "both_miss": transitions[False, False],
                                             "only_training_corpus_hit": transitions[False, True],
                                             "only_prior_full_corpus_hit": transitions[True, False]}
    assert summary["retrieval"][before]["all"]["recall_at_50"]["count"] == 311
    assert summary["retrieval"][after]["all"]["recall_at_50"]["count"] == 345
    return summary


def main(summary_path, rows_path):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    rows, qids, provenance = load_rows()
    data = summarize(rows, qids, provenance)
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    with rows_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"Saved {len(rows)} training rows, {len(qids)} fixed sample IDs, and summaries for both protocols.")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("summary", type=Path)
    cli.add_argument("rows", type=Path)
    args = cli.parse_args()
    if args.summary == args.rows or args.summary.exists() or args.rows.exists():
        cli.error("use two distinct new output paths")
    main(args.summary, args.rows)
