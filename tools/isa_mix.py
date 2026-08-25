#!/usr/bin/env python3
"""Summarize the instruction mix of one kernel in an AMDGPU assembly listing.

Kernel occupancy and roofline percentages say *that* a kernel is off its
ceiling; the instruction mix says *why*. This groups the emitted instructions of
a single kernel into the categories that matter on RDNA3 -- matrix (WMMA), VALU,
LDS, global memory, and waits/barriers -- and reports the counts so a design
change can be predicted before it is measured.

Usage:
  hipcc ... --cuda-device-only -S -o out.s src.hip
  tools/isa_mix.py out.s [kernel-name-substring] [--all]

With no substring every kernel in the file is listed by instruction count.
"""

from __future__ import annotations

import re
import sys
from collections import Counter

CATEGORIES = (
    ("matrix (wmma)", ("v_wmma",)),
    ("global load", ("global_load", "buffer_load", "flat_load")),
    ("global store", ("global_store", "buffer_store", "flat_store")),
    ("lds read", ("ds_read", "ds_load")),
    ("lds write", ("ds_write", "ds_store")),
    ("valu fp32 fma/mul/add", ("v_fma_f32", "v_fmac_f32", "v_mul_f32", "v_add_f32")),
    ("valu convert", ("v_cvt",)),
    ("valu packed", ("v_pk_", "v_dual", "v_dot")),
    ("valu other", ("v_",)),
    ("salu", ("s_add", "s_sub", "s_mul", "s_lshl", "s_and", "s_or", "s_cmp", "s_mov", "s_load")),
    ("waitcnt", ("s_waitcnt", "s_wait")),
    ("barrier", ("s_barrier",)),
    ("branch", ("s_branch", "s_cbranch")),
)


def kernels(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    # A kernel body runs from its label to the matching .Lfunc_end.
    # Labels carry a trailing "; @name" comment in clang output.
    for match in re.finditer(r"^([A-Za-z_][\w$.]*):[ \t]*(?:;[^\n]*)?\n", text, re.MULTILINE):
        name = match.group(1)
        if name.startswith(".L"):
            continue
        end = text.find(".Lfunc_end", match.end())
        if end == -1:
            continue
        body = text[match.end() : end]
        if "v_" not in body and "s_endpgm" not in body:
            continue
        out[name] = body
    return out


def mix(body: str) -> Counter:
    counts: Counter = Counter()
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith((".", ";", "//")) or line.endswith(":"):
            continue
        counts[line.split()[0]] += 1
    return counts


def report(name: str, counts: Counter) -> None:
    total = sum(counts.values())
    print(f"\n{name}")
    print(f"  {'total instructions':<26} {total:>7}")
    claimed: Counter = Counter()
    for label, prefixes in CATEGORIES:
        n = 0
        for op, c in counts.items():
            if op in claimed:
                continue
            if op.startswith(prefixes):
                n += c
                claimed[op] = 1
        if n:
            print(f"  {label:<26} {n:>7}  {100.0 * n / total:>5.1f}%")
    print("  top opcodes:")
    for op, c in counts.most_common(12):
        print(f"    {op:<28} {c:>7}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    text = open(sys.argv[1]).read()
    want = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    ks = kernels(text)
    if not ks:
        print("no kernels found")
        return 1
    if want is None:
        for name, body in sorted(ks.items(), key=lambda kv: -sum(mix(kv[1]).values()))[:20]:
            print(f"{sum(mix(body).values()):>8}  {name}")
        return 0
    hits = [(n, b) for n, b in ks.items() if want in n]
    if not hits:
        print(f"no kernel matching {want!r}; available:")
        for name in ks:
            print(f"  {name}")
        return 1
    for name, body in hits:
        report(name, mix(body))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
