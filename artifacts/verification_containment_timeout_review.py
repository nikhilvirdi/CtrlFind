"""Observe per-candidate timeout variability on a fixed subset of the training panel."""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from verification_containment import baseline, workload
import verification_completion as replay

# The saved training audit identified these sources as sensitive to the one-second cutoff.
TIMING_SENSITIVE = {"d50", "d2280", "d2312", "d2346"}


def main(output):
    saved, queries, docs, gold, examples = workload()
    plan = {qid: row for qid, row in saved["plan"].items() if TIMING_SENSITIVE.intersection(row["top"])}
    trials = []
    for workers in (2, 12, 12, 2):
        with patch.object(replay, "WORKERS", workers):
            rows = [replay.run_job(qid, row["top"], row["r"], queries, docs, gold, examples[qid])
                    for qid, row in plan.items()]
        trials.append({"workers": workers, "queries": rows})
    baseline.write_json(output, {"revision": saved["revision"], "timeout": saved["timeout"],
                                 "selection": sorted(TIMING_SENSITIVE), "trials": trials})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
