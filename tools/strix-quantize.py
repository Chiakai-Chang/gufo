#!/usr/bin/env python3
"""strix-quantize — deterministic SHQ4-T16 conversion of a source snapshot.

Implements docs/QUANTIZATION.md end-to-end pipeline steps 1 and 5 (first
slice): inspect already done by strix-inspect; here we quantize eligible
linear tensors to SHQ4-T16 U4Z G64, keep norms/embeddings/conv BF16, and write
per-tensor shards plus a reviewable quantization-plan.json.

Deterministic: identical inputs -> identical bytes.
"""
import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/strix/..")
import numpy as np
from strix import safetensors, shq
from strix.manifest import write_json


def read_bf16_tensor(path, info, dtype, shape, data_off, offset=0) -> np.ndarray:
    data = safetensors.read_tensor(path, info, dtype, shape, data_off, offset)
    # bf16 is stored as uint16 bits in the TOP 16 bits of an f32; shift left 16.
    return (np.frombuffer(data, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32).copy()


def quantize_one(name, shape, data, recipe) -> dict:
    """Returns dict with format, planes bytes, and reconstruction stats."""
    W = data.reshape(shape)
    planes = shq.quantize_shq4(W, group_size=recipe["group_size"])
    Wd = shq.dequant_shq4(planes)[: shape[0], : shape[1]]
    err = np.abs(Wd - W)
    stats = {
        "format": planes["format"],
        "max_abs_err": float(err.max()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mean_abs_err": float(err.mean()),
    }
    return planes, stats


def main(argv=None):
    ap = argparse.ArgumentParser(prog="strix-quantize")
    ap.add_argument("--source", required=True, help="HF snapshot directory")
    ap.add_argument("--out", default="artifacts/quant")
    ap.add_argument("--plan", default="artifacts/work/quantization-plan.json")
    ap.add_argument("--group-size", type=int, default=64, choices=[32, 64])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    src = Path(args.source)
    index = src / "model.safetensors.index.json"
    tensors, files = safetensors.inspect_snapshot(src, index if index.exists() else None)
    p = files[0]
    infos = safetensors.validate_file(p)
    hdr, data_off = safetensors.read_header(p)

    # Eligible linear projection weights (bulk). Everything else stays BF16.
    ELIGIBLE = (
        ".linear_attn.in_proj_qkv.weight",
        ".linear_attn.in_proj_z.weight",
        ".linear_attn.out_proj.weight",
        ".mlp.gate_proj.weight",
        ".mlp.up_proj.weight",
        ".mlp.down_proj.weight",
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
        ".self_attn.o_proj.weight",
        "mtp.fc.weight",
        "mtp.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.mlp.up_proj.weight",
        "mtp.layers.0.mlp.down_proj.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recipe = {"family": "SHQ-T16", "group_size": args.group_size,
              "variant": f"SHQ4_T16_V1_U4Z_G{args.group_size}"}
    plan = {"schema": "strix.quant-plan.v1", "recipe": recipe,
            "source_manifest": str(Path(args.source) / ".."),
            "tensors": {}}
    plan_file = Path(args.plan)

    for name in sorted(infos):
        dtype, shape, (b0, b1) = infos[name]
        if name == "__metadata__" or not name.endswith(".weight"):
            continue
        is_eligible = any(name.endswith(sfx) for sfx in ELIGIBLE)
        if is_eligible:
            if dtype != "BF16":
                plan["tensors"][name] = {"format": "BF16", "note": "source dtype not BF16"}
                continue
            data = read_bf16_tensor(p, infos[name], dtype, shape, data_off)
            planes, stats = quantize_one(name, shape, data, recipe)
            # write shard
            shard = out / (name.replace(".", "/") + ".shq4")
            shard.parent.mkdir(parents=True, exist_ok=True)
            meta = {"name": name, "shape": shape, "format": planes["format"],
                     "Np": planes["Np"], "Kp": planes["Kp"],
                     "group_size": planes["group_size"],
                     "w_len": len(planes["weight"]),
                     "s_len": len(planes["scale"]),
                     "z_len": len(planes["zero"])}
            with shard.open("wb") as f:
                f.write(json.dumps(meta).encode("utf-8"))
                f.write(b"\n")
                f.write(planes["weight"])
                f.write(planes["scale"])
                f.write(planes["zero"])
            plan["tensors"][name] = {
                "format": planes["format"],
                "shard": str(shard),
                "shape": list(shape),
                **stats,
            }
            del data
        else:
            plan["tensors"][name] = {"format": "BF16", "shard": "source-copy"}

    write_json(plan_file, plan)
    n_q = sum(1 for t in plan["tensors"].values() if t["format"].startswith("SHQ4"))
    if args.json:
        print(json.dumps({"plan": str(plan_file), "quantized_tensors": n_q,
                          "total_tensors": len(plan["tensors"])}, indent=2))
    else:
        print(f"quantization plan: {plan_file}")
        print(f"quantized {n_q} tensors to SHQ4-T16 U4Z G{args.group_size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
