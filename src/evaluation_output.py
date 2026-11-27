"""Paths for new evaluation reports and their writable embedding cache."""

import json
from pathlib import Path

from .config import ARTIFACTS, EMB_CACHE


RUN_OUTPUTS = ARTIFACTS / "evaluation_runs"


def report_path(parser, output: Path | None, filename: str) -> Path:
    path = output if output is not None else RUN_OUTPUTS / filename
    if path.is_symlink():
        parser.error(f"Output already exists: {path}. Choose a fresh path with --output.")
    path = path.resolve()
    if path.suffix.lower() != ".json":
        parser.error("The output path must end in .json.")
    if path.is_relative_to(ARTIFACTS.resolve()) and not path.is_relative_to(RUN_OUTPUTS):
        parser.error("Use artifacts/evaluation_runs/ or a path outside artifacts/ for new reports.")
    if path.exists():
        parser.error(f"Output already exists: {path}. Choose a fresh path with --output.")
    return path


def write_report(path: Path, report: dict) -> None:
    contents = json.dumps(report, indent=2, default=str)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(contents)
    except FileExistsError:
        raise SystemExit(f"Output already exists: {path}. Choose a fresh path with --output.") from None


def evaluation_encoder():
    from .encoder import E5

    if RUN_OUTPUTS.resolve() != RUN_OUTPUTS:
        raise SystemExit(f"The evaluation run directory must be a local directory: {RUN_OUTPUTS}")
    cache_path = RUN_OUTPUTS / EMB_CACHE.name
    encoder = E5(cache_path=cache_path if cache_path.exists() else EMB_CACHE)
    # Seed from the historical cache when present, then save only to the run directory.
    encoder.cache.path = cache_path
    return encoder
