"""Paths and constants shared across the pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

DATASET = "CoIR-Retrieval/apps"
MODEL_NAME = "intfloat/e5-base-v2"
MAX_SEQ_LEN = 512

QUERY_PREFIX = "query: "
DOC_PREFIX = "passage: "

# Stage 2
DEFAULT_K = 50          # candidates verified per query
EXEC_TIMEOUT = 1.0      # seconds per candidate

EMB_CACHE = ARTIFACTS / "emb_cache.npz"
EXEC_RESULTS = ARTIFACTS / "exec_results.jsonl"   # from the original full run
DOC_IDS = ARTIFACTS / "doc_ids.npy"               # maps positions in exec_results to ids
RESULTS_JSON = ARTIFACTS / "appsretrieval_results.json"

# Versions (P1 and bonus)
VERSIONS_DIR = ARTIFACTS / "versions"
VERSIONS_REPORT = ARTIFACTS / "versions_report.json"
