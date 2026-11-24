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

if os.name == "nt":
    from . import _windows_job

# Same pattern used for the reported evaluation run. Takes the first example.
EXAMPLE_RE = re.compile(
    r"-----Examples?-----\s*Input\s*\n(.*?)\n\s*Output\s*\n(.*?)(?:\n\s*Input\s*\n|\n-----|\Z)",
    re.S,
)
_OUTPUT_DECLARATION = re.compile(r"^-----Output-----[ \t]*\r?\n(.*?)(?=^-----|\Z)", re.M | re.S)
_SEPARATED_INTEGERS = re.compile(
    r"\b(?:space|whitespace)[ -]separated (?:integers?|numbers?)\b"
    r"|\b(?:separate (?:the )?(?:integers|numbers)|(?:integers|numbers) (?:are )?separated)"
    r" (?:by|with) (?:a )?(?:single )?(?:white)?spaces?\b",
    re.I,
)
_LAYOUT_REQUIRED = re.compile(
    r"\b(?:ascii|visual pattern|width|wide|padded|padding|leading|trailing|indent(?:ed|ation)?"
    r"|align(?:ed|ment)?|exactly one space"
    r"|(?:exactly|at least) \d+ (?:digits?|characters?|spaces?)|format(?:ted)? exactly"
    r"|(?:spaces?|blanks?|tabs?) (?:at the (?:start|end)|after (?:each|every|the last|the final))"
    r"|(?:start|end|finish) (?:each|every) (?:line|row) with)\b",
    re.I,
)
_INTEGER_ROW = re.compile(r"[ \t]*(?:[+-]?[0-9]+(?:[ \t]+[+-]?[0-9]+)*[ \t]*)?")


def parse_example(query: str) -> tuple[str, str] | None:
    """Return (stdin, expected_stdout) from the query, or None."""
    m = EXAMPLE_RE.search(query)
    if not m:
        return None
    inp, exp = m.group(1).strip(), m.group(2).strip()
    if not inp or not exp:
        return None
    return inp, exp


def allows_trailing_padding(query: str) -> bool:
    """Opt in only when the output section explicitly specifies separated numbers."""
    match = _OUTPUT_DECLARATION.search(query.partition("-----Example")[0])
    if match is None:
        return False
    declaration = " ".join(match.group(1).split())
    return bool(_SEPARATED_INTEGERS.search(declaration) and not _LAYOUT_REQUIRED.search(declaration))


def _trim_line_ends(output: str) -> str:
    return "\n".join(line.rstrip(" \t") for line in output.split("\n"))


def _run_candidate(args, *, input, timeout, cwd):
    if os.name == "nt":
        return _windows_job.run(args, input=input, timeout=timeout, cwd=cwd)
    return subprocess.run(args, input=input, capture_output=True, text=True, timeout=timeout, cwd=cwd)


def run_snippet(code: str, stdin_data: str, timeout: float = EXEC_TIMEOUT) -> str | None:
    """Run code with stdin_data, return stripped stdout, or None on failure."""
    try:
        with tempfile.TemporaryDirectory(prefix="cand_", ignore_cleanup_errors=True) as workdir:
            path = os.path.join(workdir, "solution.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
            r = _run_candidate(
                [sys.executable, path],
                input=stdin_data,
                timeout=timeout,
                cwd=workdir,
            )
            return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def verify(
    codes: Sequence[str],
    example: tuple[str, str],
    timeout: float = EXEC_TIMEOUT,
    workers: int | None = None,
    *,
    allow_trailing_padding: bool = False,
) -> list[bool]:
    """Run every candidate against the example. Returns a pass flag per candidate.

    Worker count defaults to the core count. More threads than cores starves
    the interpreters and turns healthy runs into timeouts.
    """
    inp, exp = example
    workers = workers or os.cpu_count() or 2
    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            outs = list(pool.map(lambda c: run_snippet(c, inp + "\n", timeout), codes))
        except BaseException:
            if os.name == "nt":
                pool.shutdown(wait=True, cancel_futures=True)
            raise
    if allow_trailing_padding and all(_INTEGER_ROW.fullmatch(line) for line in exp.split("\n")):
        exp = _trim_line_ends(exp)
        return [o is not None and _trim_line_ends(o) == exp for o in outs]
    return [o is not None and o == exp for o in outs]
