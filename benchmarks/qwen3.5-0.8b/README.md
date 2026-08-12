# Benchmark: Qwen3.5-0.8B, SHQ4-T16 U4Z G64 candidate vs bf16 teacher

Status: first vertical slice, 2026-08-11. CPU-only reference path (torch
fallback for linear attention; no flash-linear-attention/causal-conv1d fast
path). Measures OUR SHQ4 quantization quality; speed here is the reference
runtime, not Strix kernels yet.

Methodology: `docs/BENCHMARKS.md` (matched-token, per position; utilities
`strix-capture.py` / `strix-bench.py`). Raw artifacts gitignored under
`artifacts/`.

Source: `Qwen/Qwen3.5-0.8B` @ `2fc06364715b967f1860aea9cf38778875588b17`, bf16.

Candidate: all LM linear projections (qkv/z/out/mlp/full-attn/mtp) quantized
SHQ4-T16 U4Z G64; norms, embeddings, conv1d, vision kept bf16. 158 tensors
quantized, 267 MB artifact.

## Quality (candidate vs teacher, matched-token suite, 74 positions)

| metric | value |
| --- | --- |
| KL mean | 0.172 |
| KL median | 0.044 |
| KL p95 | 0.794 |
| KL p99 | 1.734 |
| KL max | 2.419 |
| top-1 agreement | 0.821 |
| teacher perplexity | 3.481 |
| candidate perplexity | 4.259 |

## Speed (CPU reference runtime)

| metric | teacher (bf16) | candidate (SHQ4 dequant bf16) |
| --- | --- | --- |
| prefill tokens/s | 99.8 | 49.1 |
| decode tokens/s | 4.09 | 4.55 |
| decode ms/token | 245 | 220 |

Note: candidate dequant-reloads to bf16 and runs through the same torch path,
so speed is the reference runtime, not our kernels. Kernel speed is a later
milestone.

## Next steps

- Port SHQ4 decode GEMV to HIP (gfx1151) and XDNA2 (AIE2P); consume the
  packed planes directly (no dequant-to-bf16).
- Imatrix / GPTQ-style scale search to pull KL tail down.
- G32 quality groups for sensitive attention tensors.
- Per-layer quality breakdown (see `docs/BENCHMARKS.md`).
