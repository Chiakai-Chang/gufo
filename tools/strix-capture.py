#!/usr/bin/env python3
"""strix-capture — capture full-precision teacher logits + perplexity.

Teacher-forced matched-token capture per docs/TESTING.md. Logits are chunked
by position and zstd-compressed under an artifact-id directory (never
committed). Writes manifest.json, tokens.u32, logits-*.f32.zst, metrics.json,
checksums.sha256.

Usage:
  strix-capture --source DIR --suite FILE --out DIR [--dtype bf16]
"""
import sys
import os
import json
import zstandard
import hashlib
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/strix/..")
import numpy as np
import torch
from strix import model as strix_model
from strix import quality


def main(argv=None):
    ap = argparse.ArgumentParser(prog="strix-capture")
    ap.add_argument("--source", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", default="artifacts/teacher")
    ap.add_argument("--model-tag", default="qwen3.5-0.8B-bf16")
    args = ap.parse_args(argv)

    suite = json.loads(Path(args.suite).read_text("utf-8"))
    model, tok = strix_model.load_teacher(args.source)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Tokenize each prompt; teacher-force next-token prediction.
    all_tokens = []
    positions = []  # (prompt_idx, start)
    logit_chunks = []
    ctx = zstandard.ZstdCompressor(level=3, write_content_size=True)

    for pi, text in enumerate(suite["prompts"]):
        ids = tok(text, return_tensors="pt")["input_ids"][0].tolist()  # [T]
        # predict next token for positions 0..T-1 (input prefix length 1..T)
        all_tokens.append(ids)
        positions.append({"prompt": pi, "offset": len(all_tokens) - 1,
                          "tokens": len(ids)})
        with torch.no_grad():
            logits = strix_model.forward_logits(model, torch.tensor([ids], dtype=torch.long))
        # logits [1, T, vocab]; we predict next token at each position using
        # full prefix up to that position — but here we use full sequence once.
        # For teacher forcing we want logits at positions 0..T-2 for targets
        # tokens[1..]. Compute per-position with full context (all but last).
        logits = logits[0].float().numpy()  # [T, V]
        # store logits for positions 0..T-1 (predict next for each)
        chunk = np.ascontiguousarray(logits, dtype=np.float32)
        logit_chunks.append(chunk.tobytes())

    tokens = np.concatenate([np.array(t, dtype=np.uint32) for t in all_tokens])
    (outdir / "tokens.u32").write_bytes(tokens.tobytes())
    with open(outdir / "positions.json", "w") as f:
        json.dump(positions, f)

    # write logit chunks (zstd), one file per prompt
    for pi, b in enumerate(logit_chunks):
        (outdir / f"logits-{pi:05d}.f32.zst").write_bytes(ctx.compress(b))
        # verify chunk length
        assert len(b) == len(all_tokens[pi]) * 248320 * 4

    # perplexity + metrics from teacher (NLL on target tokens). Compute
    # per-prompt then concatenate: rows and targets are offset differently
    # across prompts (T rows vs T-1 targets), so global indexing misaligns.
    nlls = []
    for pi, ids in enumerate(all_tokens):
        fbytes = (outdir / f"logits-{pi:05d}.f32.zst").read_bytes()
        logits = np.frombuffer(zstandard.ZstdDecompressor().decompress(fbytes, max_output_size=len(ids) * 248320 * 4), dtype=np.float32)
        logits = logits.reshape(len(ids), 248320)
        lp = quality.log_softmax(logits)
        p = np.exp(lp)
        tgt = ids[1:]
        nlls.append(-np.log(np.clip(p[np.arange(len(tgt)), tgt], 1e-30, 1.0)))
    nll = np.concatenate(nlls)
    metrics = {
        "positions": int(len(nll)),
        "perplexity": float(np.exp(nll.mean())),
        "nll_mean": float(nll.mean()),
        "prompts": len(all_tokens),
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # checksums
    lines = []
    for p in sorted(outdir.glob("*")):
        if p.name == "checksums.sha256":
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{h}  {p.name}")
    (outdir / "checksums.sha256").write_text("\n".join(lines) + "\n", "utf-8")

    manifest = {
        "schema": "strix.logit-artifact.v1",
        "model": args.model_tag,
        "vocab_size": 248320,
        "positions": positions,
        "metrics": metrics,
        "chunked": True,
    }
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps({"artifact": str(outdir), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
