"""Trusted child harness for the callable feasibility study."""
import json
import os
from pathlib import Path
import sys
from typing import List


MAX_RESULT = 32_768
SCALARS = {"NoneType": type(None), "bool": bool, "int": int, "str": str}


def encode(value, depth=0):
    if depth > 32:
        raise ValueError("value exceeds nesting limit")
    kind = type(value)
    if kind in SCALARS.values():
        return [kind.__name__, value]
    if kind is list or kind is tuple:
        return [kind.__name__, [encode(item, depth + 1) for item in value]]
    raise TypeError("unsupported return type")


def decode(encoded):
    kind, value = encoded
    if kind in SCALARS and type(value) is SCALARS[kind]:
        return value
    if kind in {"list", "tuple"}:
        items = [decode(item) for item in value]
        return tuple(items) if kind == "tuple" else items
    raise ValueError("invalid typed value")


def main(root, request):
    # Match temporary working-directory isolation while withholding caller secrets.
    environment = {k: os.environ[k] for k in ("SystemRoot", "WINDIR") if k in os.environ}
    os.environ.clear()
    os.environ.update(environment, TEMP=str(root), TMP=str(root))
    sys.path.insert(0, str(root))
    source = root / "solution.py"
    namespace = {"__name__": "candidate", "__file__": str(source)}
    if request["owner"] == "Solution":
        namespace["List"] = List
    exec(compile(source.read_bytes(), str(source), "exec", dont_inherit=True), namespace)
    name = request["name"]
    if request["owner"]:
        target = getattr(namespace[request["owner"]](), name)
    else:
        target = namespace[name]
    result = target(*decode(request["args"]))
    try:
        body = json.dumps({"value": encode(result)}, ensure_ascii=True, allow_nan=False)
        if len(body) > MAX_RESULT:
            raise ValueError("return exceeds protocol limit")
    except (TypeError, ValueError, RecursionError):
        body = json.dumps({"unsupported_value": True})
    (root / "return.json").write_text(body, encoding="ascii")


if __name__ == "__main__":
    main(Path.cwd(), json.loads(Path("request.json").read_text(encoding="ascii")))
