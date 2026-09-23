"""Content-addressed version store, modelled on how git stores files.

Every distinct snippet text is stored once as an object keyed by its hash.
A version is just a manifest mapping doc id to object hash. Committing a
new version only writes objects that did not exist before, and diffing two
versions is a comparison of hashes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class VersionStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.objects_path = self.root / "objects.jsonl"
        self.refs = self.root / "refs"
        self.log_path = self.root / "log.json"
        self._objects: dict[str, str] | None = None

    # objects -------------------------------------------------------------
    def _load(self) -> dict[str, str]:
        if self._objects is None:
            self._objects = {}
            if self.objects_path.exists():
                for line in self.objects_path.open(encoding="utf-8"):
                    rec = json.loads(line)
                    self._objects[rec["h"]] = rec["t"]
        return self._objects

    def text(self, h: str) -> str:
        return self._load()[h]

    # versions ------------------------------------------------------------
    def exists(self) -> bool:
        return self.log_path.exists()

    def names(self) -> list[str]:
        return json.loads(self.log_path.read_text()) if self.log_path.exists() else []

    def manifest(self, name: str) -> dict[str, str]:
        return json.loads((self.refs / f"{name}.json").read_text())["docs"]

    def checkout(self, name: str) -> dict[str, str]:
        objs = self._load()
        return {d: objs[h] for d, h in self.manifest(name).items()}

    def commit(self, name: str, docs: dict[str, str]) -> dict:
        """Store a version. Returns counts of new objects and changes vs the parent."""
        objs = self._load()
        self.refs.mkdir(parents=True, exist_ok=True)
        new_objects = 0
        manifest = {}
        with self.objects_path.open("a", encoding="utf-8") as f:
            for d, t in docs.items():
                h = content_hash(t)
                if h not in objs:
                    objs[h] = t
                    f.write(json.dumps({"h": h, "t": t}) + "\n")
                    new_objects += 1
                manifest[d] = h

        log = self.names()
        parent = log[-1] if log else None
        (self.refs / f"{name}.json").write_text(
            json.dumps({"name": name, "parent": parent, "docs": manifest}))
        if name not in log:
            log.append(name)
        self.log_path.write_text(json.dumps(log))

        stats = {"version": name, "docs": len(manifest), "new_objects": new_objects}
        if parent:
            stats.update(self.diff(parent, name))
        return stats

    def diff(self, a: str, b: str) -> dict:
        ma, mb = self.manifest(a), self.manifest(b)
        return {
            "added": sum(1 for d in mb if d not in ma),
            "removed": sum(1 for d in ma if d not in mb),
            "changed": sum(1 for d in mb if d in ma and ma[d] != mb[d]),
        }
