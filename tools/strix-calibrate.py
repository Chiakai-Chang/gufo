#!/usr/bin/env python3
"""strix-calibrate — capture per-input-channel imatrix (E[x^2]) for SHQ4 tensors.

Runs the calibration suite through the full-precision teacher and records, for
every eligible LM linear projection, its per-input-channel importance vector
h[j] = E[x_j^2] over the calibration tokens. This is the activation-weighted
reconstruction objective used by the imatrix/GPTQ-style scale search in
strix-quantize (--imatrix). Mirror of strix-capture's suite/hash discipline.

Layout of the output artifact dir:
  manifest.json        schema, hashes (source + dataset), token count, per
                       tensor { file, dim, count, mean_channel_energy }
  <tensor>.f32         one importance vector per eligible tensor (float32, K)

Usage:
  strix-calibrate --source DIR --suite FILE --out DIR
"""
import sys
import os
import json
import hashlib
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/strix/..")
import numpy as np
import torch
from strix import model as strix_model


# Eligible module suffixes = weight-tensor suffixes minus ".weight".
_ELIGIBLE_MODULES = tuple(s[: -len(".weight")] for s in strix_model.ELIGIBLE)


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="strix-calibrate")
    ap.add_argument("--source", required=True, help="HF snapshot directory")
    ap.add_argument("--suite", required=True, help="calibration suite JSON")
    ap.add_argument("--out", default="artifacts/calib")
    args = ap.parse_args(argv)

    src = Path(args.source)
    suite = json.loads(Path(args.suite).read_text("utf-8"))
    model, tok = strix_model.load_teacher(str(src))

    # Collect target modules: language_model projections only (mtp head is not
    # loaded by this model adapter, so it is excluded and falls back to range
    # scaling in strix-quantize).
    targets = {}
    for name, mod in model.named_modules():
        if not name.startswith("language_model."):
            continue
        if any(name.endswith(sfx) for sfx in _ELIGIBLE_MODULES):
            targets[name] = mod
    if not targets:
        raise SystemExit("no eligible modules found")

    acc = {name: (np.zeros(mod.weight.shape[1], dtype=np.float64), 0)
           for name, mod in targets.items()}

    def make_hook(name):
        def hook(mod, args, kwargs=None):
            x = args[0] if args else None
            if x is None and kwargs:
                x = next(iter(kwargs.values()))
            xt = x.detach().float()
            x2 = xt.square().sum(dim=tuple(range(xt.dim() - 1))).numpy()
            s, n = acc[name]
            s += x2
            n += int(xt.shape[:-1].numel())
            acc[name] = (s, n)
            return None
        return hook

    handles = []
    for name, mod in targets.items():
        handles.append(mod.register_forward_hook(make_hook(name)))

    total_tokens = 0
    try:
        with torch.no_grad():
            for text in suite["prompts"]:
                ids = tok(text, return_tensors="pt")["input_ids"]
                model.language_model(input_ids=ids, use_cache=False)
                total_tokens += int(ids.shape[1])
    finally:
        for h in handles:
            h.remove()

    # Normalize to per-channel second moments and write artifact.
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for name in sorted(targets):
        s, n = acc[name]
        imp = (s / n).astype(np.float32) if n > 0 else np.ones(targets[name].weight.shape[1], np.float32)
        fname = name.replace("model.", "") + ".f32"
        (outdir / fname).write_bytes(imp.tobytes())
        tensors[name] = {
            "file": fname,
            "dim": int(targets[name].weight.shape[1]),
            "count": int(n),
            "mean_channel_energy": float(imp.mean()),
        }

    # Hashes: source snapshot + dataset.
    source_hash = hashlib.sha256()
    src_files = sorted(src.glob("model.*.safetensors")) + sorted(src.glob("config.json"))
    for p in src_files:
        source_hash.update(p.name.encode())
        source_hash.update(hash_file(p).encode())
    suite_hash = hash_file(Path(args.suite))

    manifest = {
        "schema": "strix.imatrix.v1",
        "model": suite.get("model", "unknown"),
        "source": str(src),
        "source_hash": source_hash.hexdigest(),
        "dataset": str(Path(args.suite)),
        "dataset_hash": suite_hash,
        "count": int(total_tokens),
        "tensors": tensors,
    }
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps({"artifact": str(outdir), "modules": len(tensors),
                      "tokens": total_tokens, "schema": manifest["schema"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())