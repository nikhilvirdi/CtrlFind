"""Stage 2: verify candidate snippets by running them on the query's example.

Competitive programming problems ship an example input and its expected
output. A snippet that reproduces the expected output is almost certainly
a solution to that problem, so passing snippets are moved to the top.

Each candidate runs in its own interpreter process, in a throwaway working
directory, with a hard timeout. Inside Docker the container adds a further
layer of isolation.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from .config import EXEC_TIMEOUT

# Same pattern used for the reported evaluation run. Takes the first example.
EXAMPLE_RE = re.compile(
    r"-----Examples?-----\s*Input\s*\n(.*?)\n\s*Output\s*\n(.*?)(?:\n\s*Input\s*\n|\n-----|\Z)",
    re.S,
)


def parse_example(query: str) -> tuple[str, str] | None:
    """Return (stdin, expected_stdout) from the query, or None."""
    m = EXAMPLE_RE.search(query)
    if not m:
        return None
    inp, exp = m.group(1).strip(), m.group(2).strip()
    if not inp or not exp:
        return None
    return inp, exp


def run_snippet(code: str, stdin_data: str, timeout: float = EXEC_TIMEOUT) -> str | None:
    """Run code with stdin_data, return stripped stdout, or None on failure."""
    workdir = tempfile.mkdtemp(prefix="cand_")
    path = os.path.join(workdir, "solution.py")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        r = subprocess.run(
            [sys.executable, path],
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
        )
        return r.stdout.strip()
    except Exception:
        return None
    finally:
        try:
            os.remove(path)
            os.rmdir(workdir)
        except OSError:
            pass


def verify(
    codes: Sequence[str],
    example: tuple[str, str],
    timeout: float = EXEC_TIMEOUT,
    workers: int | None = None,
) -> list[bool]:
    """Run every candidate against the example. Returns a pass flag per candidate.

    Worker count defaults to the core count. More threads than cores starves
    the interpreters and turns healthy runs into timeouts.
    """
    inp, exp = example
    workers = workers or os.cpu_count() or 2
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outs = list(pool.map(lambda c: run_snippet(c, inp + "\n", timeout), codes))
    return [o is not None and o == exp for o in outs]
