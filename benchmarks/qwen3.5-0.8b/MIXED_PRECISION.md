# Mixed-Precision Experiment: Qwen3.5-0.8B (SHQ-T16 tiers)

Status: 2026-08-13. Compares SHQ-T16 mixed-precision recipes against the
bf16 teacher, same matched-token suite as the base benchmark. Same source
revision, tokenizer, imatrix (`artifacts/calib`, 666-token disjoint set).

Recipes are per-tensor tiers (never per-block). Tiers used:

| encoding | bpw | role |
|---|---|---|
| SHQ4-G64-U4Z | 4.50 | bulk linear tensors |
| SHQ4-G32-U4Z | 4.625 | sensitive attention (evaluated, no gain) |
| SHQ8-G64 | 8.25 | high-precision tier (embed, ffn_down, linear-attn) |
| BF16 | 16 | norms, conv1d, small projections, vision |

**True model size** = quantized shard bytes + BF16-kept tensor bytes (source
bf16). Shard-only totals undercount BF16-kept tensors (embed is 508 MB bf16).

## Results (KL = matched-token mean; suite 78 positions)

| recipe | true MB | S8 | G32 | G64 | KL mean | KL p95 | top1 | ppl |
|---|---|---|---|---|---|---|---|---|
| bulk_g64 (baseline) | 990.7 | 0 | 0 | 158 | 0.1154 | 0.434 | 0.833 | 3.882 |
| ffn_only | 1035.9 | 25 | 0 | 133 | 0.0862 | 0.309 | 0.885 | 3.956 |
| embed_only | 744.4 | 1 | 0 | 158 | 0.1189 | 0.434 | 0.833 | 3.890 |
| embed_ffn | 789.5 | 26 | 0 | 133 | 0.0866 | 0.308 | 0.872 | 3.957 |
| embed_attn | 746.4 | 1 | 28 | 130 | 0.1232 | 0.407 | 0.859 | 3.903 |
| mirror_no_lin | 791.5 | 26 | 28 | 105 | 0.0887 | 0.297 | 0.859 | 3.987 |
| unsloth_mirror | 884.4 | 80 | 28 | 51 | 0.0383 | 0.156 | 0.910 | 3.831 |
| full_shq8 | 999.1 | 159 | 0 | 0 | 0.0013 | 0.003 | 0.974 | 3.453 |

## What the ladder shows (isolated levers)

1. **ffn_down SHQ8 is the quality lever.** `ffn_only` drops KL 0.1154 -> 0.0862
   (-26%) for upcasting only 25 tensors (one ffn_down per layer). This is the
   single highest quality-per-byte move, and it mirrors unsloth's `Q6_K` ffn_down.
2. **embed SHQ8 is a free size cut.** `embed_only` is quality-neutral (KL 0.1189,
   within suite noise of 0.1154) but is 246 MB smaller than baseline. `embed_ffn`
   (789.5 MB) matches `ffn_only` (1035.9 MB) quality at 246 MB smaller — the embed
   upcast pays for itself entirely. **The bf16 embed policy was over-spending the
   largest tensor for zero measured quality.**
3. **G32 attention adds nothing.** `mirror_no_lin` (adds G32 attention) is slightly
   worse than `embed_ffn` (0.0887 vs 0.0866); `embed_attn` (G32 only) is worse than
   `embed_only`. Drop G32: it buys no quality and adds a second group-size kernel
   path.
4. **linear_attn SHQ8 is the biggest further lever.** `unsloth_mirror` (adds
   linear_attn qkv/z/out SHQ8) drops KL 0.0866 -> 0.0383 (-56%) for +95 MB. Mirrors
   unsloth's Q8_0/Q5_K linear-attn. Best quality-per-MB overall.
5. `full_shq8` (KL 0.0013) is the ceiling but is bigger than baseline — not a
   deployment size.

## Chosen deployment recipe: `embed_ffn`

Embed SHQ8 + ffn_down SHQ8 + bulk SHQ4 G64; norms/conv/vision bf16.
789.5 MB, KL 0.0866. Best balance of quality, size, and hardware decode cost.

Quality variant: `unsloth_mirror` (884 MB, KL 0.0383) when prefill-heavy or
quality floor demands, accepting linear_attn SHQ8 decode bandwidth.

## Strix Halo hardware compatibility (format-level)

Measured kernel speed is **not** available yet — `src/main.cpp` is only a
device probe; SHQ4/SHQ8 decode GEMV kernels are the next milestone. What is
verified at the format level:

- SHQ8-T16 uses the **identical T16 tile layout and packing order** as SHQ4-T16
  (verified byte-identical to the reference loop; conformance suite passes).
  The decode GEMV kernel is the same structure — only the per-weight read width
  differs (1 byte vs nibble). So a SHQ8 tensor needs no new kernel family.
- Precision is **per-tensor**, never per-block (docs/QUANTIZATION.md), so hot
  kernels do not branch per block.
- The chosen recipe keeps the bulk hot path on a single SHQ4-G64 group size
  (G32 dropped), and the SHQ8 tier is small (embed + ffn_down only): on the
  single-token decode GEMV path SHQ8 costs 2x memory bandwidth per weight vs
  SHQ4, and decode is bandwidth-bound, so keeping the tier small protects
  decode throughput. Prefill (compute-bound matmul) is far less sensitive.

## Tooling

`tools/strix/recipe.py` defines presets; `strix-quantize.py --recipe <name>`
converts; `strix-mp-experiment.py` quantizes + benches each preset and prints
the comparison table. Plans in `artifacts/work/plan-<preset>.json`,
shards in `artifacts/quant-<preset>/`.
