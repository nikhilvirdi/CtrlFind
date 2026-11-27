"""Check consequential non-gold passes on independently derived training fixtures."""

import argparse
import hashlib
from math import isqrt
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

import retrieval_consistency as audit
from src import execute
from src.config import EXEC_TIMEOUT


CASES = {
    "q1606": ("2 2", "8", ["d5451", "d6198", "d7643", "d5746"],
              "For remainder 1 and quotient 1 or 2, the only nice integers are 3 and 5."),
    "q2155": ("2 2\n1\n2", "2\n2", ["d5865", "d6516", "d6270", "d6093"],
              "Neither threshold is attained after one draw. Two draws have probability 1/2 of including both types."),
    "q2323": ("4\n0 0\n0 1\n1 0\n1 1\n2\n0 1 1 2\n2 5 5 8",
              "4 0 0 0 1 1 0 1 1\n4 -1 -1 -1 2 2 -1 2 2", ["d7089"],
              "Independently enumerated integer points whose multiset of distances matches the published query."),
}


def geolocation_fixture():
    antennas = [(0, 0), (0, 1), (1, 0), (1, 1)]
    lines = []
    for distances in ([0, 1, 1, 2], [2, 5, 5, 8]):
        limit = isqrt(max(distances))
        matches = [(x, y) for x in range(-limit, limit + 1) for y in range(-limit, limit + 1)
                   if sorted((x - a)**2 + (y - b)**2 for a, b in antennas) == distances]
        lines.append(" ".join(map(str, [len(matches), *(n for point in matches for n in point)])))
    return "\n".join(lines)


def inspect(code, stdin):
    with TemporaryDirectory(prefix="ctrlfind_witness_") as root:
        path = Path(root) / "solution.py"
        path.write_text(code, encoding="utf-8")
        try:
            result = execute._run_candidate([sys.executable, str(path)], input=stdin + "\n",
                                            timeout=EXEC_TIMEOUT, cwd=root)
        except TimeoutError as exc:
            return {"cleanup_error": repr(exc)}
        except OSError as exc:
            return {"infrastructure_error": repr(exc)}
        except subprocess.TimeoutExpired:
            return {"timeout": True}
        stderr = result.stderr
        if stderr is not None:
            stderr = stderr.replace(root, "<candidate-workdir>")
        return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": stderr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    output = parser.parse_args().output
    assert not output.exists()
    assert geolocation_fixture() == CASES["q2323"][1]
    queries = audit.pq.read_table(audit.parquet("queries/*.parquet"), columns=["_id", "text"],
                                  filters=[("partition", "=", "train"), ("_id", "in", list(CASES))])
    ids = [did for _, _, candidates, _ in CASES.values() for did in candidates] + ["d2323"]
    docs = audit.pq.read_table(audit.parquet("corpus/*.parquet"), columns=["_id", "text"],
                               filters=[("_id", "in", ids)])
    snippets = {row["_id"]: row["text"] for row in docs.to_pylist()}
    try:
        compile(snippets["d2323"], "d2323", "exec")
    except SyntaxError as exc:
        gold = {"line": exc.lineno, "error": exc.msg}
    else:
        gold = {"compiles": True}
    receipt = {"dataset_revision": audit.REVISION, "timeout": EXEC_TIMEOUT, "q2323_gold_compile": gold, "queries": {}}
    for row in queries.to_pylist():
        qid = row["_id"]
        stdin, expected, candidates, derivation = CASES[qid]
        published = re.findall(r"\bInput\s*\n(.*?)\n\s*Output\s*\n(.*?)(?=\n\s*Input\s*\n|\n-----|\Z)",
                               row["text"].partition("-----Example")[2], re.S)
        assert (stdin.split(), expected.split()) in [(a.split(), b.split()) for a, b in published], qid
        first_input, first_expected = execute.parse_example(row["text"])
        results = {}
        for did in candidates:
            code = snippets[did]
            first, followup = inspect(code, first_input), inspect(code, stdin)
            original_pass = first.get("exit_code") == 0 and (first.get("stdout") or "").strip() == first_expected
            definitive_failure = followup.get("exit_code") not in (0, None) or (
                followup.get("exit_code") == 0 and followup.get("stdout") is not None and
                followup["stdout"].split() != expected.split()
            )
            results[did] = {"code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
                            "first_example_pass": original_pass, "first": first, "followup": followup,
                            "label": "confirmed_incorrect" if original_pass and definitive_failure else "unknown"}
        receipt["queries"][qid] = {"first_input": first_input, "first_expected": first_expected,
                                   "next_input": stdin, "next_expected": expected,
                                   "independent_derivation": derivation, "candidates": results}
    audit.write_json(output, receipt)


if __name__ == "__main__":
    main()
