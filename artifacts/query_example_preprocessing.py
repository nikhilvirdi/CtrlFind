"""Conservative ablation of explicitly labelled numeric examples for E5."""
from __future__ import annotations

import re

_HEADING = re.compile(r"^-----([a-z][a-z ]*):?-----[ \t]*(?:\r?\n|\Z)", re.I | re.M)
_LABEL = re.compile(r"^[ \t]*(input|output):?[ \t]*(?:\r?\n|\Z)", re.I | re.M)
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:e[+-]?[0-9]+)?", re.I)
_SEPARATE = {"example input": "example output", "sample input": "sample output"}


def _data(text, *, output=False):
    tokens = text.split()
    return bool(tokens) and all(_NUMBER.fullmatch(t) or (output and t.casefold() in {"yes", "no"})
                                for t in tokens)


def _combined(text):
    labels = list(_LABEL.finditer(text))
    if not labels or text[:labels[0].start()].strip():
        return False
    if [m[1].lower() for m in labels] != ["input", "output"] * (len(labels) // 2):
        return False
    ends = [m.start() for m in labels[1:]] + [len(text)]
    return all(_data(text[label.end():end], output=label[1].lower() == "output")
               for label, end in zip(labels, ends))


def strip_examples(query):
    """Drop recognized sections only. Preserve all bytes outside those sections."""
    headings = list(_HEADING.finditer(query))
    sections = [(m[1].strip().lower(), m.start(), m.end()) for m in headings]
    ends = [s[1] for s in sections[1:]] + [len(query)]
    remove = []
    i = 0
    while i < len(sections):
        name, start, body = sections[i]
        if name in {"example", "examples"} and _combined(query[body:ends[i]]):
            remove.append((start, ends[i]))
        elif name in _SEPARATE and i + 1 < len(sections):
            following, _, next_body = sections[i + 1]
            if (following == _SEPARATE[name] and _data(query[body:ends[i]])
                    and _data(query[next_body:ends[i + 1]], output=True)):
                remove.append((start, ends[i + 1]))
                i += 1
        i += 1

    for start, end in reversed(remove):
        query = query[:start] + query[end:]
    return query
