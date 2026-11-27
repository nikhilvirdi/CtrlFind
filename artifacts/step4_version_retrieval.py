"""Training-only Step 4, using the unchanged Windows verifier and version index."""
import argparse
import ast
from collections import Counter
import math
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import tempfile
import time
import warnings

import verification_depth as depth
import verification_depth_v3 as v3

from src.cache import EmbeddingCache
from src.config import DEFAULT_K, MODEL_NAME
from src.evolution import History, rank_verified
from src.history import reformat
from src.versions import VersionStore, content_hash


audit, np, execute = depth.audit, depth.np, depth.execute
REF = "352cc1c"
ADD, SUB = ast.Add, ast.Sub
SWAPS = {ADD: SUB, SUB: ADD, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult, ast.Mod: ast.FloorDiv}
FLIPS = {ast.Lt: ast.GtE, ast.GtE: ast.Lt, ast.LtE: ast.Gt, ast.Gt: ast.LtE, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
FAMILIES = ("arithmetic", "comparison", "off_by_one", "constant")
RTOL, ATOL = 1e-4, 2e-6


def integrity():
    files = ("src/_windows_job.py", "src/execute.py", "src/config.py", "src/encoder.py",
             "src/cache.py", "src/evolution.py", "src/versions.py", "src/history.py", "index.py")
    changed = subprocess.check_output(["git", "diff", REF, "--name-only", "--", *files], cwd=audit.ROOT)
    assert not changed, changed
    receipt = audit.read_json(audit.HERE / "windows_cleanup_k50_validation_train.json")
    assert receipt["source_commit"].startswith(REF) and receipt["regressions"]["count"] == 41
    assert receipt["regressions"]["exit_code"] == 0
    assert all(v["counts"]["attempts"] == 10_000 and v["counts"]["verdict_changes"] == 0
               for v in receipt["live_vs_controls"].values())
    for name in ("verifier_sha256", "windows_job_sha256"):
        assert depth.runtime()[name] == receipt["runtime"][name]
    frozen = v3.check_frozen()
    saved = audit.read_json(audit.HERE / "verification_depth_v3_post_stop_audit.json")
    assert frozen == saved["frozen_reference"]
    for name, digest in saved["archived_artifact_sha256"].items():
        assert audit.sha_file(v3.RUN / name) == digest
    return frozen, {name: audit.sha_file(audit.ROOT / name) for name in files}


def inputs():
    queries, sources, gold, _, dev, _, _ = audit.inputs()
    qids = sorted(q for q in dev if sources[q] == depth.SOURCE)
    saved = audit.read_json(v3.RUN / "plan.json")
    assert len(qids) == 477 and set(qids) == set(saved["rows"])
    assert all(audit.sha_bytes(queries[q].encode("utf-8")) == saved["rows"][q]["query_sha256"] for q in qids)
    source = audit.parquet("corpus/*.parquet")
    rows = audit.pq.read_table(source, columns=["_id", "text"], filters=[("partition", "=", "train")]).to_pylist()
    base = dict(sorted((row["_id"], row["text"]) for row in rows))
    assert len(base) == len(rows) == 5000 and base.keys() == set(gold.values())
    return base, queries, gold, qids


def sites(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and type(node.op) in SWAPS:
            yield "arithmetic", node, "op", None
        if isinstance(node, ast.Compare):
            for index, op in enumerate(node.ops):
                if type(op) in FLIPS:
                    yield "comparison", node, "ops", index
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range" and node.args:
            index = min(1, len(node.args) - 1)
            if not isinstance(node.args[index], ast.Starred):
                yield "off_by_one", node, "args", index
        if isinstance(node, ast.Subscript) and not isinstance(node.slice, (ast.Slice, ast.Tuple, ast.Starred)):
            yield "off_by_one", node, "slice", None
        if isinstance(node, ast.Constant) and type(node.value) is int:
            yield "constant", node, "value", None


def span(node):
    return (node.lineno, node.col_offset), (node.end_lineno, node.end_col_offset)


def choose_edits(tree, qid):
    rng = random.Random(f"{audit.SEED}:mutate:{qid}")
    families = list(FAMILIES)
    rng.shuffle(families)
    indexed = list(enumerate(sites(tree)))
    chosen, used = [], []
    for family in families:
        candidates = [(index, node) for index, (kind, node, _, _) in indexed if kind == family]
        rng.shuffle(candidates)
        for index, node in candidates:
            left, right = span(node)
            if any(left < end and start < right for start, end in used):
                continue
            chosen.append({"site": index, "family": family, "start": left, "end": right,
                           "delta": rng.choice((-1, 1)) if family == "off_by_one" else None})
            used.append((left, right))
            break
        if len(chosen) == 3:
            break
    return chosen


def mutate(code, edits):
    tree = ast.parse(code)
    indexed = list(sites(tree))
    for edit in edits:
        kind, node, field, index = indexed[edit["site"]]
        assert edit["family"] == kind
        old = getattr(node, field) if index is None else getattr(node, field)[index]
        if kind == "arithmetic":
            new = SWAPS[type(old)]()
        elif kind == "comparison":
            new = FLIPS[type(old)]()
        elif kind == "constant":
            new = old + 1
        else:
            op = ADD() if edit["delta"] == 1 else SUB()
            new = ast.BinOp(left=old, op=op, right=ast.Constant(1))
        if index is None:
            setattr(node, field, new)
        else:
            getattr(node, field)[index] = new
    result = ast.unparse(tree)
    compile(result, "<mutant>", "exec")
    return result


def manifest_versions(base, queries, gold, qids):
    versions = {name: base.copy() for name in ("v1", "v2", "v3")}
    rows = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        for q in qids:
            did = gold[q]
            example = execute.parse_example(queries[q])
            row = {"gold": did, "example": example, "padding": execute.allows_trailing_padding(queries[q])}
            rows[q] = row
            if example is None:
                row["status"] = "no_example"
                continue
            try:
                compile(base[did], "<original>", "exec")
                formatted = reformat(base[did])
                if formatted is None or ast.dump(ast.parse(formatted)) != ast.dump(ast.parse(base[did])):
                    row["status"] = "reformat_ast_mismatch"
                    continue
                compile(formatted, "<reformat>", "exec")
            except (SyntaxError, ValueError, RecursionError) as error:
                row.update(status="original_unavailable", error=str(error))
                continue
            for snapshot in versions.values():
                snapshot[did] = formatted
            edits = choose_edits(ast.parse(formatted), q)
            row["edits"] = edits
            if not edits:
                row["status"] = "no_site"
                continue
            try:
                mutant = mutate(formatted, edits)
            except (SyntaxError, ValueError, RecursionError) as error:
                row.update(status="noncompiling_mutant", error=str(error))
                continue
            cut = (len(edits) + 1) // 2 if len(edits) > 1 else random.Random(f"{audit.SEED}:repair:{q}").randrange(2)
            repaired = mutate(formatted, edits[cut:]) if edits[cut:] else formatted
            assert mutant != formatted
            versions["v2"][did], versions["v3"][did] = mutant, repaired
            row.update(status="retained", reverted_sites=[edit["site"] for edit in edits[:cut]],
                       versions={name: content_hash(snapshot[did]) for name, snapshot in versions.items()})
    return versions, rows


def publish_plan(folder, versions, rows, qids, frozen, source_hashes, base):
    manifests = {name: {did: content_hash(code) for did, code in version.items()}
                 for name, version in versions.items()}
    text_changes = {name: {did: code for did, code in version.items() if code != base[did]}
                    for name, version in versions.items()}
    scope = random.Random(f"{audit.SEED}:scope").sample(qids, 200)
    benchmark = sorted(random.Random(f"{audit.SEED}:index").sample(list(base), 500))
    changes = {str(pct): sorted(random.Random(f"{audit.SEED}:index:{pct}").sample(benchmark, 5 * pct))
               for pct in (1, 10, 50)}
    protocol = (audit.ROOT / "docs" / "research-log.md").read_text(encoding="utf-8").split(
        "### Version retrieval preregistration\n", 1)[1].split("\n### ", 1)[0].strip()
    assert "nt" == depth.os.name and DEFAULT_K == 50 and depth.EXEC_TIMEOUT == 1.0
    audit.write_json(folder / "plan.json", {
        "protocol_sha256": depth.digest(protocol), "seed": audit.SEED, "dataset_revision": audit.REVISION,
        "training_query_ids": qids, "scoped_query_ids": scope, "benchmark_ids": benchmark, "benchmark_changes": changes,
        "inputs": {pattern: audit.sha_file(audit.parquet(pattern)) for pattern in
                   ("queries/*.parquet", "corpus/*.parquet", "data/train-*.parquet")},
        "runtime": depth.runtime(), "torch_threads": audit.torch.get_num_threads(),
        "packages": audit.package_versions(), "production_source_sha256": source_hashes,
        "frozen": frozen, "models": audit.model_files(), "model_revision": audit.MODEL_REVISION,
        "allclose": {"rtol": RTOL, "atol": ATOL}, "rows": rows, "manifests": manifests, "text_changes": text_changes,
    })
    return scope, benchmark, changes


def make_store(directory, versions):
    store = VersionStore(directory)
    for name, docs in versions.items():
        store.commit(name, docs)
    return store


def history_index(store, model, cache_path):
    start = time.perf_counter()
    model.cache = EmbeddingCache(cache_path, MODEL_NAME)
    index = History(store, model)
    return index, {"seconds": time.perf_counter() - start, "encoded": int(index.encoded)}


def version_rows(history, name):
    return [(history.entries[i].doc_id, content_hash(history.entries[i].text)) for i in history.rows[name]]


def structural(scratch, model, versions):
    cache = scratch / "incremental.npz"
    store = make_store(scratch / "versions", {"v1": versions["v1"]})
    _, seed = history_index(store, model, cache)
    store.commit("v2", versions["v2"])
    inc, inc_timing = history_index(store, model, cache)
    full, full_timing = history_index(make_store(scratch / "fresh_v2", {"v2": versions["v2"]}),
                                     model, scratch / "fresh_v2.npz")
    expected = [(did, content_hash(code)) for did, code in versions["v2"].items()]
    match = version_rows(inc, "v2") == expected == version_rows(full, "v2")
    a, b = inc.emb[inc.rows["v2"]], full.emb[full.rows["v2"]]
    equivalent = bool(np.allclose(a, b, rtol=RTOL, atol=ATOL))
    result = {"rows": len(expected), "ordered_ids_and_content_match": match, "allclose": equivalent,
              "maximum_absolute_difference": float(np.max(np.abs(a-b))), "seed": seed,
              "incremental": inc_timing, "fresh": full_timing}
    store.commit("v3", versions["v3"])
    return store, cache, result


def benchmark(scratch, model, base, ids, all_changes):
    old = {did: base[did] for did in ids}
    seed = scratch / "benchmark_seed.npz"
    history_index(make_store(scratch / "benchmark_old", {"v1": old}), model, seed)
    observations = {}
    for pct, changed in all_changes.items():
        updated = {**old, **{did: old[did] + f"\n# step4 index {pct} {did}\n" for did in changed}}
        full_store = make_store(scratch / f"full_{pct}", {"v2": updated})
        inc_store = make_store(scratch / f"inc_{pct}", {"v1": old, "v2": updated})
        pairs = []
        for repeat in range(3):
            cache = {arm: scratch / f"bench_{pct}_{repeat}_{arm}.npz" for arm in ("full", "incremental")}
            shutil.copyfile(seed, cache["incremental"])
            order = ("full", "incremental") if repeat % 2 == 0 else ("incremental", "full")
            result = {"order": order}
            for arm in order:
                store = full_store if arm == "full" else inc_store
                _, result[arm] = history_index(store, model, cache[arm])
            pairs.append(result)
            print(f"P1 {pct}% repeat {repeat+1}: {result}", flush=True)
        observations[pct] = pairs
    return observations


class Verdicts:
    def __init__(self):
        self.records = {}

    def run(self, codes, row):
        fixture = (*row["example"], row["padding"])
        keys = {name: (*fixture, content_hash(code)) for name, code in codes.items()}
        missing = {}
        for name, key in keys.items():
            if key not in self.records and key not in missing:
                missing[key] = name
        run_ids = list(missing.values())
        if run_ids:
            batch = depth.observe(run_ids, codes, row)
            for result in batch["records"]:
                self.records[keys[result["id"]]] = {k: v for k, v in result.items() if k not in ("id", "stderr")}
        return {name: self.records[key] for name, key in keys.items()}, run_ids


def classify(versions, rows, cache):
    labels, primary = {}, []
    for qid, row in rows.items():
        if row["status"] != "retained":
            continue
        family = {name: version[row["gold"]] for name, version in versions.items()}
        observed, launches = cache.run(family, row)
        killed = not observed["v2"]["passed"]
        labels[qid] = {"status": "KILLED" if killed else "SURVIVOR", "versions": observed, "actual_runs": launches}
        if killed and observed["v1"]["passed"]:
            primary.append(qid)
        if len(labels) % 75 == 0:
            print(f"Classified {len(labels)} mutants; {len(primary)} primary problems", flush=True)
    return labels, primary


def scoped_retrieval(h, versions, gold, vectors, scope):
    details = {}
    for name, docs in versions.items():
        rows = []
        for q in scope:
            order, _ = h.search(vectors[q], name, DEFAULT_K)
            entries = [h.entries[i] for i in order]
            found = [(entry.doc_id, content_hash(entry.text)) for entry in entries]
            leaks = [did for did, digest in found if did not in docs or digest != content_hash(docs[did])]
            rank = next((i for i, entry in enumerate(entries, 1) if entry.doc_id == gold[q]), None)
            rows.append({"query": q, "rank": rank, "hits": found, "leaks": leaks,
                         "duplicates": len(found) - len(set(found)), "missing": DEFAULT_K - len(found),
                         "shared_hits": sum(len(entry.versions) > 1 for entry in entries)})
        details[name] = rows
    return details


def quality(order, passing, failing):
    ranks = [rank for rank, i in enumerate(order, 1) if i in passing]
    highest = ranks[0]
    worst_competitor = next(rank for rank, i in enumerate(order, 1) if i in failing)
    dcg = sum(1 / math.log2(rank + 1) for rank in ranks if rank <= 10)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(10, len(ranks)) + 1))
    return {"passing_rank": highest, "first_failing_rank": worst_competitor, "passing_ranks": ranks,
            "working_first": int(highest < worst_competitor), "ndcg": dcg / ideal,
            "mrr": 1 / highest if highest <= 10 else 0, "relevant_top10": [int(i) for i in order[:10] if i in passing]}


def bonus(history, rows, labels, vectors, primary, cache):
    evaluations, noise = {}, {}
    repeat = set(random.Random(f"{audit.SEED}:noise").sample(primary, min(24, len(primary))))
    doc_ids = [entry.doc_id for entry in history.entries]
    families = {}
    for index, did in enumerate(doc_ids):
        families.setdefault(did, []).append(index)
    for q in primary:
        row = rows[q]
        order, scores = history.search(vectors[q], None, len(history.entries))
        top = order[:DEFAULT_K]
        family = families[row["gold"]]
        # The family labels identify relevance even below the execution budget.
        records, _ = cache.run({i: history.entries[i].text for i in family}, row)
        assert all(records[i]["passed"] == labels[q]["versions"][name]["passed"]
                   for i in family for name in history.entries[i].versions)
        passing = {i for i, record in records.items() if record["passed"]}
        failing = set(family) - passing
        assert passing and failing
        verdicts, launches = cache.run({i: history.entries[i].text for i in top}, row)
        ranked = rank_verified(order, scores, {i: r["passed"] for i, r in verdicts.items()})
        result = {"similarity": quality(order, passing, failing), "execution": quality(ranked, passing, failing),
                  "top50": top, "reranked_top50": ranked[:DEFAULT_K], "family": records, "verdicts": verdicts,
                  "actual_runs": launches, "sibling_slots": sum(doc_ids[i] == row["gold"] for i in top)}
        evaluations[q] = result
        if q in repeat and launches:
            # The original scoring cache remains untouched.
            repro, _ = Verdicts().run({i: history.entries[i].text for i in launches}, row)
            changed = [{"id": i, "previous": verdicts[i], "repeat": repro[i]} for i in launches
                       if verdicts[i]["passed"] != repro[i]["passed"] or
                       (verdicts[i]["phase"] == "timeout") != (repro[i]["phase"] == "timeout")]
            repeated_flags = {i: r["passed"] for i, r in {**verdicts, **repro}.items()}
            repeated = quality(rank_verified(order, scores, repeated_flags), passing, failing)
            noise[q] = {"attempts": len(launches), "changes": changed,
                        "original": result["execution"], "repeat": repeated}
        if len(evaluations) % 25 == 0:
            print(f"Bonus {len(evaluations)}/{len(primary)}", flush=True)
    catalog = [{"doc_id": entry.doc_id, "content": content_hash(entry.text), "versions": entry.versions}
               for entry in history.entries]
    return evaluations, noise, catalog


def bootstrap(evaluations):
    metrics = ("working_first", "ndcg", "mrr")
    if not evaluations:
        return {}, {}
    draw = np.random.default_rng(audit.SEED).integers(0, len(evaluations), size=(10_000, len(evaluations)))
    summary, samples = {}, {}
    for metric in metrics:
        a, b = (np.asarray([row[arm][metric] for row in evaluations.values()], dtype="float64")
                for arm in ("similarity", "execution"))
        summary[metric] = {}
        for name, values in (("similarity", a), ("execution", b), ("delta", b - a)):
            sampled = values[draw].mean(axis=1)
            samples[f"{metric}_{name}"] = sampled
            summary[metric][name] = {"mean": float(values.mean()), "ci95": np.percentile(sampled, [2.5, 97.5]).tolist()}
    return summary, samples


def main(folder):
    assert not folder.exists() and __debug__
    frozen, source_hashes = integrity()
    audit.configure()
    base, queries, gold, qids = inputs()
    versions, rows = manifest_versions(base, queries, gold, qids)
    folder.mkdir()
    scope, bench_ids, changes = publish_plan(folder, versions, rows, qids, frozen, source_hashes, base)
    plan_sha = audit.sha_file(folder / "plan.json")
    cache = Verdicts()
    labels, primary = classify(versions, rows, cache)
    audit.write_json(folder / "mutant_verdicts.json", {"primary": primary, "labels": labels})
    print(f"Mutation phase complete: {len(labels)} retained, {len(primary)} primary KILLED problems", flush=True)

    with tempfile.TemporaryDirectory(prefix="ctrlfind-step4-") as temporary:
        scratch = Path(temporary)
        model = audit.load_model(scratch / "model_warmup.npz")
        model.encode_docs(list(base.values())[:8])
        print("Starting full-manifest structural comparison", flush=True)
        store, path, structure = structural(scratch, model, versions)
        audit.write_json(folder / "structure.json", structure)
        history, _ = history_index(store, model, path)
        print("Starting cold index timing grid", flush=True)
        timings = benchmark(scratch, model, base, bench_ids, changes)
        audit.write_json(folder / "timings.json", timings)
        # The saved query vectors are exclusively from the validated training census.
        dev = audit.read_json(depth.BASELINE)["config"]["query_ids"]
        query_matrix, query_manifest = audit.read_vectors(audit.HERE / "retrieval_consistency_run" / "dev_vectors.npz",
                                                       dev, queries)
        vectors = dict(zip(dev, query_matrix, strict=True))
        scoped = scoped_retrieval(history, versions, gold, vectors, scope)
        audit.write_json(folder / "scoped.json", scoped)
        evaluations, noise, catalog = bonus(history, rows, labels, vectors, primary, cache)

    quality_metrics, samples = bootstrap(evaluations)
    audit.write_json(folder / "bonus.json", {"catalog": catalog, "rows": evaluations, "noise": noise})
    with (folder / "bootstrap.npz").open("xb") as target:
        np.savez_compressed(target, **samples)
    assert audit.sha_file(folder / "plan.json") == plan_sha
    assert integrity() == (frozen, source_hashes)
    gate = {
        "p1_structure": structure["ordered_ids_and_content_match"] and structure["allclose"],
        "p1_timing10": statistics.median(x["incremental"]["seconds"] for x in timings["10"]) <
                       statistics.median(x["full"]["seconds"] for x in timings["10"]),
        "p1_scope": all(not row["leaks"] and not row["duplicates"] and not row["missing"]
                        for cases in scoped.values() for row in cases),
        "bonus_population": len(primary) >= 100,
        "bonus_working_first": len(primary) >= 100 and quality_metrics["working_first"]["delta"]["ci95"][0] > 0,
    }
    scoped_metrics = {name: {"queries": len(cases), "hits": sum(len(r["hits"]) for r in cases),
                             "leaks": sum(len(r["leaks"]) for r in cases),
                             "duplicates": sum(r["duplicates"] for r in cases), "missing": sum(r["missing"] for r in cases),
                             "shared_hits": sum(r["shared_hits"] for r in cases),
                             "recall_at_10": sum(r["rank"] is not None and r["rank"] <= 10 for r in cases)/len(cases),
                             "ndcg_at_10": sum(1/math.log2(r["rank"]+1) for r in cases
                                              if r["rank"] is not None and r["rank"] <= 10)/len(cases)}
                      for name, cases in scoped.items()}
    launch_count = sum(len(x["actual_runs"]) for x in evaluations.values())
    phases = Counter(r["phase"] for e in evaluations.values() for r in e["verdicts"].values())
    repeat_changes = [c for n in noise.values() for c in n["changes"]]
    results = {
        "gates": {key: "PASSED" if value else "FAILED" for key, value in gate.items()},
        "plan_sha256": plan_sha, "query_vectors_sha256": query_manifest["vector_sha256"],
        "selection": dict(Counter(row["status"] for row in rows.values())),
        "mutants": dict(Counter(v["status"] for v in labels.values())),
        "killed_phases": dict(Counter(v["versions"]["v2"]["phase"] for v in labels.values() if v["status"] == "KILLED")),
        "primary": len(primary), "scoped": scoped_metrics, "version_entries": len(catalog), "quality": quality_metrics,
        "verification": {"slots": len(primary)*DEFAULT_K, "runs": launch_count,
                         "reused": len(primary)*DEFAULT_K - launch_count, "slot_phases": dict(phases),
                         "non_gold_passes": sum(r["passed"] and catalog[i]["doc_id"] != rows[q]["gold"]
                                                for q, e in evaluations.items() for i, r in e["verdicts"].items()),
                         "slots_passing": sum(r["passed"] for e in evaluations.values() for r in e["verdicts"].values()),
                         "sibling_slots": sum(e["sibling_slots"] for e in evaluations.values())},
        "repeat": {"queries": len(noise), "attempts": sum(n["attempts"] for n in noise.values()),
                   "verdict_changes": sum(c["previous"]["passed"] != c["repeat"]["passed"] for c in repeat_changes),
                   "timeout_changes": sum((c["previous"]["phase"] == "timeout") != (c["repeat"]["phase"] == "timeout")
                                          for c in repeat_changes),
                   "queries_with_metric_changes": [q for q, n in noise.items() if n["original"] != n["repeat"]]},
    }
    audit.write_json(folder / "results.json", results)
    print(results["gates"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
