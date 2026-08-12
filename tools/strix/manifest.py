"""Manifest helpers (source-manifest.json, quantization-plan.json)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


def sha256_text(s) -> str:
    data = s if isinstance(s, bytes) else s.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, obj, indent=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, sort_keys=True) + "\n", "utf-8")
    return path


def write_source_manifest(out: Path, *, repository, revision, architecture,
                          source_storage_dtype, files: list, tokenizer_sha,
                          template_sha):
    manifest = {
        "schema": "strix.source.v1",
        "repository": repository,
        "revision": revision,
        "architecture": architecture,
        "source_storage_dtype": source_storage_dtype,
        "tokenizer_sha256": tokenizer_sha,
        "chat_template_sha256": template_sha,
        "files": files,
    }
    write_json(out, manifest)
    return out
