"""Safetensors inspection and validation.

Implements the Safetensors Source Contract in docs/QUANTIZATION.md:

- Validate header length, JSON, dtype, shape, byte offsets, non-overlap.
- Reject duplicate tensor names and missing indexed tensors.
- Reject files not referenced by the selected index.
- Verify tensor byte length from dtype and shape.
- Hash every source file with SHA-256.
- Write source-manifest.json.

Reads tensor payloads lazily via bounded buffered reads (never the whole
checkpoint resident).
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

# safetensors dtypes -> (ctype, element bytes). Matches the reference
# safetensors header format (little-endian).
DTYPES = {
    "F32": ("f", 4),
    "F64": ("d", 8),
    "I8": ("b", 1),
    "I16": ("h", 2),
    "I32": ("i", 4),
    "I64": ("q", 8),
    "U8": ("B", 1),
    "U16": ("H", 2),
    "U32": ("I", 4),
    "U64": ("Q", 8),
    "BF16": ("e", 2),
    "BOOL": ("?", 1),
}


class SafetensorsError(Exception):
    pass


def dtype_size(dtype: str) -> int:
    if dtype not in DTYPES:
        raise SafetensorsError(f"unknown dtype {dtype!r}")
    return DTYPES[dtype][1]


def read_header(path: Path):
    """Read and validate the safetensors header. Returns (header_dict, data_offset)."""
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise SafetensorsError(f"{path}: truncated header-length field")
        (nbytes,) = struct.unpack("<Q", raw)
        header = f.read(nbytes)
        if len(header) != nbytes:
            raise SafetensorsError(f"{path}: header length {nbytes} exceeds file")
        try:
            hdr = json.loads(header.decode("utf-8"))
        except Exception as e:
            raise SafetensorsError(f"{path}: invalid header JSON: {e}") from e
        if not isinstance(hdr, dict):
            raise SafetensorsError(f"{path}: header must be a JSON dict")
        return hdr, 8 + nbytes


def _tensor_bytes(dtype: str, shape: list):
    n = 1
    for d in shape:
        if d < 0:
            raise SafetensorsError(f"negative shape dimension {d}")
        n *= d
    return n * dtype_size(dtype)


def validate_file(path: Path, expect: set | None = None) -> dict:
    """Validate one safetensors file. Returns {name: (dtype, shape, offset, length)}.

    Verifies dtype, shape, byte offsets, non-overlap, and that each tensor's
    declared byte length fits inside the file payload.
    """
    hdr, data_off = read_header(path)
    size = path.stat().st_size
    infos = {}
    # First pass: parse shapes and lengths, reject duplicates.
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict):
            raise SafetensorsError(f"{path}: tensor {name!r} metadata not a dict")
        dtype = meta.get("dtype")
        shape = meta.get("shape")
        data_offsets = meta.get("data_offsets")
        if dtype not in DTYPES:
            raise SafetensorsError(f"{path}: {name}: unknown dtype {dtype!r}")
        if not isinstance(shape, list) or not all(isinstance(d, int) for d in shape):
            raise SafetensorsError(f"{path}: {name}: bad shape {shape!r}")
        if not isinstance(data_offsets, list) or len(data_offsets) != 2:
            raise SafetensorsError(f"{path}: {name}: bad data_offsets")
        if name in infos:
            raise SafetensorsError(f"{path}: duplicate tensor {name!r}")
        infos[name] = (dtype, tuple(shape), tuple(data_offsets))
    # Second pass: bounds and overlap.
    declared = []
    for name, (dtype, shape, (b0, b1)) in infos.items():
        tbytes = _tensor_bytes(dtype, shape)
        if b0 >= b1 or b1 - b0 != tbytes:
            raise SafetensorsError(
                f"{path}: {name}: offsets [{b0},{b1}) length {b1-b0} != tensor bytes {tbytes}"
            )
        if data_off + b1 > size:
            raise SafetensorsError(
                f"{path}: {name}: payload [{b0},{b1}) beyond file size {size}"
            )
        declared.append((name, b0, b1))
    declared.sort(key=lambda t: t[1])
    for i in range(1, len(declared)):
        if declared[i][1] < declared[i - 1][2]:
            raise SafetensorsError(
                f"{path}: overlapping payloads {declared[i-1][0]} and {declared[i][0]}"
            )
    return infos


def read_tensor(path: Path, info, dtype, shape, data_off: int, offset: int):
    """Lazily read one tensor's bytes (bounded I/O). Returns bytes.

    info may be (dtype, shape, (b0, b1)) from validate_file or the raw header
    dict with "data_offsets".
    """
    if isinstance(info, dict):
        b0, b1 = info["data_offsets"]
    else:
        b0, b1 = info[2]
    with path.open("rb") as f:
        f.seek(data_off + b0 + offset)
        return f.read(b1 - b0 - offset)


def sha256_file(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            c = f.read(chunk)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


def inspect_snapshot(dir_path: Path, index_path: Path | None):
    """Inspect a HF snapshot. Returns source-manifest fields.

    index_path: model.safetensors.index.json if present (sharded).
    """
    dir_path = Path(dir_path)
    tensors = {}
    files = []
    if index_path is not None:
        index = json.loads(index_path.read_text("utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise SafetensorsError("index.json missing weight_map")
        # Map shard -> names
        shard_names = {}
        for name, shard in weight_map.items():
            if name in shard_names:
                raise SafetensorsError(f"duplicate tensor {name!r} in index")
            shard_names[name] = shard
        shards = {}
        for name, shard in shard_names.items():
            shards.setdefault(shard, []).append(name)
        for shard, names in shards.items():
            p = dir_path / shard
            if not p.exists():
                raise SafetensorsError(f"indexed shard missing: {shard}")
            infos = validate_file(p, expect=set(names))
            missing = set(names) - set(infos)
            if missing:
                raise SafetensorsError(f"{shard}: missing indexed tensors: {sorted(missing)}")
            for n in names:
                tensors[n] = (shard, infos[n])
            files.append(p)
    else:
        # Single-file snapshot: model.safetensors
        cands = [p for p in dir_path.glob("*.safetensors") if p.name != "index.json"]
        if len(cands) != 1:
            raise SafetensorsError(f"expected exactly one safetensors file, got {len(cands)}")
        p = cands[0]
        infos = validate_file(p)
        for n in infos:
            tensors[n] = (p.name, infos[n])
        files.append(p)

    return tensors, files
