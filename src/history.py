"""Synthetic version history for a corpus that has none.

CoIR APPS is a single snapshot, so versions are generated:

  v1  the original snippets
  v2  some snippets reformatted (behaviour unchanged), some given a small bug
  v3  some v2 bugs fixed, some fresh bugs introduced, some snippets reformatted

A bug is one operator or constant changed, like `<` to `<=` or `n` to `n + 1`.
The text barely changes, which is exactly what makes versions hard to tell
apart by similarity.

For the tracked snippets, whether the newest version is buggy is a coin
flip. Otherwise "always pick the newest version" would look perfect.

A bug that leaves the output unchanged on every available test cannot be
told apart from a correct version by anyone, so for tracked snippets only
bugs that change the output on the problem's example are kept. This is the
usual mutation-testing rule of discarding mutants the tests cannot see.
"""
from __future__ import annotations

import ast
import random
import warnings
from typing import Callable

SWAPS = {
    ast.Add: ast.Sub, ast.Sub: ast.Add,
    ast.Lt: ast.LtE, ast.LtE: ast.Lt,
    ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
}


def _parse(code: str) -> ast.Module:
    # dataset snippets contain regex strings like "\d" that trigger harmless warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(code)


def reformat(code: str) -> str | None:
    """Same behaviour, different text: normalised layout, comments dropped."""
    try:
        out = ast.unparse(_parse(code))
    except Exception:
        return None
    return out if out.strip() != code.strip() else "# tidied\n" + out


def inject_bug(code: str, rng: random.Random) -> str | None:
    """Change one operator or integer constant. None if nothing to change."""
    try:
        tree = _parse(code)
    except Exception:
        return None
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and type(node.op) in SWAPS:
            sites.append(("binop", node))
        elif isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                if type(op) in SWAPS:
                    sites.append(("cmp", (node, i)))
        elif isinstance(node, ast.Constant) and type(node.value) is int:
            sites.append(("const", node))
    if not sites:
        return None
    kind, target = rng.choice(sites)
    if kind == "binop":
        target.op = SWAPS[type(target.op)]()
    elif kind == "cmp":
        node, i = target
        node.ops[i] = SWAPS[type(node.ops[i])]()
    else:
        target.value = target.value + 1
    try:
        return ast.unparse(tree)
    except Exception:
        return None


def make_history(
    base: dict[str, str],
    tracked: set[str],
    seed: int = 0,
    reformat_frac: float = 0.08,
    bug_frac: float = 0.04,
    observable: Callable[[str, str], bool] | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, bool]]]:
    """Return ({version: {doc_id: text}}, {version: {doc_id: is_buggy}}).

    Tracked snippets always get a distinct version in v2 and v3, with the
    newest version buggy half the time. Others change at the given rates.
    """
    rng = random.Random(seed)
    ids = list(base)

    def tracked_bug(d: str) -> str | None:
        # observable(d, code) says whether code behaves differently on d's example
        for _ in range(10):
            c = inject_bug(base[d], rng)
            if c and (observable is None or observable(d, c)):
                return c
        return None
    v1, v2, v3 = dict(base), dict(base), {}
    bug = {"v1": {d: False for d in ids}, "v2": {}, "v3": {}}

    # v2
    for d in ids:
        r = rng.random()
        if d in tracked:
            want_bug = rng.random() < 0.5
        else:
            want_bug = r < bug_frac
        changed = None
        if want_bug:
            changed = tracked_bug(d) if d in tracked else inject_bug(base[d], rng)
            if changed:
                v2[d], bug["v2"][d] = changed, True
                continue
        if d in tracked or r < bug_frac + reformat_frac:
            changed = reformat(base[d])
            if changed:
                v2[d] = changed
        bug["v2"][d] = False

    # v3
    for d in ids:
        was_bug = bug["v2"][d]
        r = rng.random()
        if d in tracked:
            want_bug = rng.random() < 0.5
            if want_bug:
                if was_bug:
                    v3[d], bug["v3"][d] = v2[d], True           # bug survives
                    continue
                changed = tracked_bug(d)
                if changed:
                    v3[d], bug["v3"][d] = changed, True
                    continue
            fixed = reformat(base[d])                           # fixed and tidied
            v3[d] = (fixed + "\n# v3\n") if fixed else base[d] + "\n# v3\n"
            bug["v3"][d] = False
            continue
        if was_bug and r < 0.5:                                 # half the bugs fixed
            v3[d], bug["v3"][d] = reformat(base[d]) or base[d], False
        elif not was_bug and r < bug_frac:
            changed = inject_bug(base[d], rng)
            v3[d], bug["v3"][d] = (changed, True) if changed else (v2[d], False)
        else:
            v3[d], bug["v3"][d] = v2[d], was_bug

    return {"v1": v1, "v2": v2, "v3": v3}, bug