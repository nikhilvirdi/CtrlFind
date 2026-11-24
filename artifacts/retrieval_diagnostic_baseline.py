"""Run evaluate.py --mode cached with an isolated, cache-only MTEB result."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
CACHE_RESULT = Path("results/prism__e5-base-v2-exec-rerank/1/AppsRetrieval.json")


def main(output):
    cache = Path(os.environ.get("MTEB_CACHE", Path.home() / ".cache/mteb")) / CACHE_RESULT
    frozen = ROOT / "artifacts/baseline_frozen.json"
    with TemporaryDirectory(prefix="ctrlfind_cached_") as folder:
        scratch = Path(folder)
        shutil.copy2(ROOT / "evaluate.py", scratch)
        shutil.copytree(ROOT / "src", scratch / "src", ignore=shutil.ignore_patterns("__pycache__"))
        inputs = scratch / "artifacts"
        inputs.mkdir()
        for name in ("doc_ids.npy", "exec_results.jsonl", "emb_cache.npz"):
            shutil.copy2(ROOT / "artifacts" / name, inputs)
        cached_copy = scratch / "mteb" / CACHE_RESULT
        cached_copy.parent.mkdir(parents=True)
        shutil.copy2(cache, cached_copy)
        # A cache miss must fail before MTEB can load any evaluation data.
        (scratch / "sitecustomize.py").write_text(
            "try:\n"
            "    from functools import partial\n"
            "    import mteb\n"
            "    mteb.evaluate = partial(mteb.evaluate, overwrite_strategy='only-cache')\n"
            "except Exception as error:\n"
            "    raise SystemExit(f'Cannot install cache-only guard: {error}')\n",
            encoding="utf-8",
        )
        command = [sys.executable, "evaluate.py", "--mode", "cached"]
        completed = subprocess.run(
            command, cwd=scratch, check=True, capture_output=True, text=True, encoding="utf-8",
            env={**os.environ, "PYTHONPATH": str(scratch), "MTEB_CACHE": str(scratch / "mteb"),
                 "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "PYTHONUTF8": "1",
                 "PYTHONDONTWRITEBYTECODE": "1"},
        )
        print(completed.stdout, end="")
        result = (inputs / "appsretrieval_results.json").read_bytes()
        score = json.loads(result)["scores"]["test"][0]
        summary = {
            "command": "python evaluate.py --mode cached",
            "mteb_overwrite_strategy": "only-cache",
            "cached_result_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
            "replayed_result_sha256": hashlib.sha256(result).hexdigest(),
            "byte_identical_to_frozen": result == frozen.read_bytes(),
            "ndcg_at_10": score["ndcg_at_10"],
            "mrr_at_10": score["mrr_at_10"],
        }
    with Path(output).open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main(sys.argv[1])
