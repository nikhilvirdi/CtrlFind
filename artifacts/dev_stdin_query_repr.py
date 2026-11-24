"""Run the preregistered dev-stdin representation experiment on A or the locked winner on B."""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import time

import numpy as np
import torch
from mteb.similarity_functions import cos_sim

import retrieval_consistency as audit
from dev_stdin_full_baseline import VECTORS, calculate_metrics
from src.config import MAX_SEQ_LEN, QUERY_PREFIX


ARMS = ("V1", "V2")
HEAD = 320
STRIDE = 256
RESAMPLES = 10_000
TIMED_QUERIES = 32
PREREG_TITLE = "### Query representation preregistration"
INPUTS = (
    audit.HERE / "dev_stdin_ab_split.json",
    audit.HERE / "dev_stdin_full_baseline.json",
    audit.HERE / "dev_stdin_train.json",
)


class TokenEncoder:
    def __init__(self, e5):
        self.e5 = e5
        self.tokenizer = e5.model.tokenizer
        self.prefix = self.tokenizer.encode(QUERY_PREFIX, add_special_tokens=False)
        self.special = self.tokenizer.num_special_tokens_to_add(pair=False)
        self.budget = MAX_SEQ_LEN - len(self.prefix) - self.special
        self.cls, self.sep = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        assert len(self.prefix) == self.special == 2 and self.budget == 508
        assert self.tokenizer.encode(QUERY_PREFIX, add_special_tokens=True) == self.window([])
        assert self.tokenizer.padding_side == self.tokenizer.truncation_side == "right"
        e5.model.eval()

    def body(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)

    def window(self, body):
        return [self.cls, *self.prefix, *body, self.sep]

    def plan(self, bodies, arm, skip_first=False):
        owners, offsets, sequences = [], [], []
        for i, body in enumerate(bodies):
            assert len(body) > self.budget
            if arm == "V1":
                tail = self.budget - HEAD
                windows = [(0, body[:HEAD] + body[-tail:])]
            else:
                assert arm == "V2"
                n = (len(body) - self.budget + STRIDE - 1) // STRIDE
                windows = [(j * STRIDE, body[j * STRIDE:j * STRIDE + self.budget])
                           for j in range(int(skip_first), n + 1)]
            for offset, window in windows:
                owners.append(i)
                offsets.append(offset)
                sequences.append(self.window(window))
        return owners, offsets, sequences

    @torch.inference_mode()
    def embed(self, sequences, label=None):
        order = sorted(range(len(sequences)), key=lambda i: -len(sequences[i]))
        encoded = np.empty((len(sequences), 768), dtype=np.float32)
        for n in range(0, len(order), audit.BATCH):
            batch = order[n:n + audit.BATCH]
            ids = [sequences[i] for i in batch]
            width = len(ids[0])
            pad = self.tokenizer.pad_token_id
            input_ids = torch.tensor([x + [pad] * (width - len(x)) for x in ids])
            features = {"input_ids": input_ids, "token_type_ids": torch.zeros_like(input_ids),
                        "attention_mask": torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids])}
            output = self.e5.model(features)["sentence_embedding"]
            encoded[batch] = torch.nn.functional.normalize(output, p=2, dim=1).numpy()
            if label and (n // audit.BATCH % 5 == 4 or n + audit.BATCH >= len(order)):
                print(f"{label} encoded {min(len(order), n + audit.BATCH)}/{len(order)}", flush=True)
        audit.validate_vectors(encoded, len(sequences))
        return encoded

    def calibrate(self, texts, bodies):
        prefix_texts = [QUERY_PREFIX + text for text in texts]
        production = self.e5.model[0].preprocess(prefix_texts)["input_ids"].tolist()
        c0 = [self.window(body[:self.budget]) for body in bodies]
        assert c0 == production
        direct = self.embed(c0)
        public = self.e5._encode_raw(prefix_texts)
        np.testing.assert_allclose(direct, public, rtol=0, atol=1e-6)
        return {"queries": len(c0), "input_ids_match": True,
                "max_abs_embedding_difference": float(np.max(np.abs(direct - public)))}


def collapse(scores, owners, n):
    result = np.full((n, scores.shape[1]), -np.inf, dtype=scores.dtype)
    np.maximum.at(result, owners, scores)
    assert np.isfinite(result).all()
    return result


def frozen_shape_cosine(vectors, docs, height, positions):
    # Matching the archived matrix shape preserves float32 ordering at very close scores.
    batch = np.repeat(vectors[:1], height, axis=0)
    batch[positions] = vectors
    return cos_sim(batch, docs).numpy()[positions]


def time_arms(half, arms, ids, queries, encoder, docs):
    fixed = random.Random(f"{audit.SEED}:latency:{half}").sample(sorted(ids), TIMED_QUERIES)
    order = ("C0", *arms)
    times, windows = {}, {}
    for start, sequence in ((0, order), (audit.BATCH, order[::-1])):
        texts = [queries[qid] for qid in fixed[start:start + audit.BATCH]]
        for arm in sequence:
            before = time.perf_counter()
            if arm == "C0":
                vectors = encoder.e5._encode_raw([QUERY_PREFIX + text for text in texts])
                result = cos_sim(vectors, docs).numpy()
                n_windows = len(vectors)
            else:
                parts = [encoder.body(text) for text in texts]
                owners, _, inputs = encoder.plan(parts, arm)
                vectors = encoder.embed(inputs)
                result = collapse(cos_sim(vectors, docs).numpy(), owners, len(texts))
                n_windows = len(inputs)
            elapsed = time.perf_counter() - before
            assert result.shape == (len(texts), len(docs))
            times.setdefault(arm, []).append(elapsed)
            windows.setdefault(arm, []).append(n_windows)
            print(f"{half} timing batch {start // audit.BATCH + 1} {arm} {elapsed:.3f} s", flush=True)
    means = {arm: sum(series) / TIMED_QUERIES for arm, series in times.items()}
    return {"query_ids": fixed, "batch_orders": [list(order), list(order[::-1])],
            "batch_wall_seconds": times, "windows_by_batch": windows, "seconds_per_query": means,
            "increase_vs_c0": {arm: means[arm] - means["C0"] for arm in arms}}


def paired_metrics(ids, before, after):
    pre = [before[qid]["rank"] for qid in ids]
    post = [after[qid]["rank"] for qid in ids]
    changes = {k: {"entered": [qid for qid, a, b in zip(ids, pre, post, strict=True) if b <= k < a],
                   "left": [qid for qid, a, b in zip(ids, pre, post, strict=True) if a <= k < b]}
               for k in audit.CUTOFFS}
    a, b = calculate_metrics(before[qid] for qid in ids), calculate_metrics(after[qid] for qid in ids)
    return {
        "queries": len(ids), "C0": a, "variant": b,
        "recall_delta_pp": {k: 100 * (len(c["entered"]) - len(c["left"])) / len(ids) for k, c in changes.items()},
        "ndcg_delta_pp": b["ndcg_at_10"]["percent"] - a["ndcg_at_10"]["percent"],
        "rank_improved": sum(b < a for a, b in zip(pre, post, strict=True)),
        "rank_regressed": sum(b > a for a, b in zip(pre, post, strict=True)),
        "rank_unchanged": sum(b == a for a, b in zip(pre, post, strict=True)),
        "cutoff_transitions": changes,
    }


def summarize(half, ids, truncated, sources, baseline, changed):
    result = {}
    after = {**baseline, **changed}
    for name, ids in (("full_half", ids), ("truncated", truncated)):
        result[name] = paired_metrics(ids, baseline, after)
        result[name]["sources"] = {
            source: paired_metrics([qid for qid in ids if sources[qid] == source], baseline, after)
            for source in sorted(audit.DEV_SOURCES)
        }
    deltas = np.array([int(after[qid]["rank"] <= 50) - int(baseline[qid]["rank"] <= 50) for qid in truncated],
                      dtype=np.int8)
    draw = np.random.default_rng(audit.SEED).integers(0, len(deltas), size=(RESAMPLES, len(deltas)))
    ci = (100 * np.quantile(deltas[draw].mean(axis=1), [0.025, 0.975])).tolist()
    result["truncated"]["recall_at_50_paired_bootstrap"] = {
        "ci_95_pp": ci, "resamples": RESAMPLES, "seed": audit.SEED, "unit": "query",
    }
    result["gates_excluding_latency"] = {
        "truncated_recall_50": result["truncated"]["recall_delta_pp"][50] >= (3 if half == "A" else 1.5)
                                and (ci[0] > 0 if half == "A" else True),
        "full_half_ndcg_10": result["full_half"]["ndcg_delta_pp"] >= 0,
        "source_recall_50": all(x["recall_delta_pp"][50] >= -2 for x in result["truncated"]["sources"].values()
                                if x["queries"] >= 100),
    }
    return result


def preregistration():
    source = (audit.ROOT / "docs" / "research-log.md").read_text(encoding="utf-8")
    text = source.split(PREREG_TITLE + "\n", 1)[1].split("\n### ", 1)[0].strip()
    assert all(item in text for item in ("2026-11-25 02:22:06 +05:30", "3.0 percentage points",
                                        "1.5 percentage points", "10,000", "0.1 seconds/query"))
    return {"section": PREREG_TITLE, "sha256": audit.sha_bytes(text.encode("utf-8"))}


def write_vectors(path, qids, owners, offsets, vectors):
    with path.open("xb") as out:
        np.savez_compressed(out, query_ids=qids, owners=owners, text_offsets=offsets, vectors=vectors)


def run_half(output, half):
    registration = preregistration()
    script_sha = audit.sha_file(Path(__file__))
    metadata_path, baseline_path, census_path = INPUTS
    saved_split, frozen, census = map(audit.read_json, INPUTS)
    metadata, base = saved_split["halves"][half], frozen["rows"]
    all_ids = metadata["query_ids"]
    selected = [qid for qid in all_ids if census["rows"][qid]["formatted_tokens"] > MAX_SEQ_LEN]
    same_inputs = {"split": audit.sha_file(metadata_path), "baseline": audit.sha_file(baseline_path),
                   "census": audit.sha_file(census_path)}
    args = {"registration": registration, "input_sha256": same_inputs, "runner_sha256": script_sha}
    if half == "A":
        arms = ARMS
        assert not any(output.glob("A*")) and not (output / "winner.json").exists()
    else:
        lock = audit.read_json(output / "winner.json")
        assert lock["inputs"] == args and lock["A_results_sha256"] == audit.sha_file(output / "A_results.json")
        arms = (lock["arm"],)
        assert arms[0] in ARMS
        audit.write_json(output / "B_started.json", {"locked_arm": arms[0], "inputs": args})

    queries, sources, gold, _, dev, old, ablation = audit.inputs()
    docs, _, _, _, _, passages, _, _ = audit.corpus(gold, old, ablation)
    assert docs == frozen["config"]["candidate_corpus"]["document_ids_in_parquet_order"]
    assert audit.vector_digest(passages) == frozen["config"]["candidate_corpus"]["vector_sha256"]
    query_vectors, manifest = audit.read_vectors(VECTORS, dev, queries)
    assert manifest == frozen["config"]["query_vector_manifest"]
    positions = {qid: i for i, qid in enumerate(dev)}
    selected_positions = [positions[qid] for qid in selected]
    c0 = frozen_shape_cosine(query_vectors[selected_positions], passages, len(dev), selected_positions)
    c0_ranks = audit.fixed_ranks(c0, docs, selected, gold)
    assert all(r["rank"] == base[qid]["rank"] and r["top200"] == base[qid]["top200"] for qid, r in c0_ranks.items())

    encoder = TokenEncoder(audit.load_model(output / "unused.npz"))
    texts = [queries[qid] for qid in selected]
    bodies = [encoder.body(text) for text in texts]
    for qid, text, body in zip(selected, texts, bodies, strict=True):
        production = encoder.tokenizer.encode(QUERY_PREFIX + text, add_special_tokens=False, verbose=False)
        assert production == encoder.prefix + body
        assert len(production) + encoder.special == census["rows"][qid]["formatted_tokens"]
    calibration = encoder.calibrate(texts[:4], bodies[:4])
    print(f"{half} locked population {len(all_ids)}; truncated {len(selected)}; arms {arms}", flush=True)
    summaries = {}
    for arm in arms:
        owners, offsets, inputs = encoder.plan(bodies, arm, skip_first=arm == "V2")
        start = time.perf_counter()
        encoded = encoder.embed(inputs, f"{half} {arm}")
        forward_seconds = time.perf_counter() - start
        vector_path = output / f"{half}_{arm}_vectors.npz"
        write_vectors(vector_path, selected, owners, offsets, encoded)
        slots = list(range(len(encoded))) if arm == "V2" else selected_positions
        scores = frozen_shape_cosine(encoded, passages, len(dev), slots)
        if arm == "V2":
            scores = np.maximum(c0, collapse(scores, owners, len(selected)))
        else:
            assert owners == list(range(len(selected)))
        rows = audit.fixed_ranks(scores, docs, selected, gold)
        for qid, row in rows.items():
            row.update(C0_rank=base[qid]["rank"], formatted_tokens=census["rows"][qid]["formatted_tokens"],
                       source=sources[qid])
        metrics = summarize(half, all_ids, selected, sources, base, rows)
        summary = {"measures": metrics, "encoding": {"new_windows": len(inputs), "forward_wall_seconds": forward_seconds,
                                                       "vectors_sha256": audit.vector_digest(encoded)},
                   "rows": rows}
        audit.write_json(output / f"{half}_{arm}_rankings.json", summary)
        summaries[arm] = {k: v for k, v in summary.items() if k != "rows"}
        print(f"{half} {arm} saved scores", flush=True)

    timings = time_arms(half, arms, selected, queries, encoder, passages)
    for arm in arms:
        gates = dict(summaries[arm]["measures"]["gates_excluding_latency"])
        gates["query_side_latency"] = timings["increase_vs_c0"][arm] <= 0.1
        summaries[arm]["gates"] = gates
        summaries[arm]["passed"] = all(gates.values())

    viable = [arm for arm in arms if summaries[arm]["passed"]]
    decision, winner = "REJECTED", None
    if half == "A" and viable:
        order = sorted(viable, key=lambda arm: -summaries[arm]["measures"]["truncated"]["variant"]["recall_at_50"]["count"])
        if len(order) == 2 and (summaries[order[0]]["measures"]["truncated"]["variant"]["recall_at_50"]["count"] ==
                                summaries[order[1]]["measures"]["truncated"]["variant"]["recall_at_50"]["count"]):
            decision = "A_TIE_B_CLOSED"
        else:
            winner, decision = order[0], "LOCKED_FOR_B"
    elif half == "B" and viable:
        winner, decision = viable[0], "KEEP"
    results_path = output / f"{half}_results.json"
    audit.write_json(results_path, {"inputs": args, "half": half, "query_ids": all_ids, "truncated": selected,
                                    "calibration": calibration, "head": HEAD, "stride": STRIDE,
                                    "text_token_budget": encoder.budget, "timings": timings,
                                    "arms": summaries, "decision": decision, "winner": winner})
    if half == "A" and winner:
        audit.write_json(output / "winner.json", {"arm": winner, "inputs": args,
                                                  "A_results_sha256": audit.sha_file(results_path)})
    print(half, decision, winner, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("half", choices=("A", "B"))
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    audit.configure()
    args.output.mkdir(parents=True, exist_ok=True)
    run_half(args.output, args.half)
