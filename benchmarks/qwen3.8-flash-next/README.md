# Qwen3.8-Flash-Next on Strix Halo

Linux x86-64, AMD Strix Halo `gfx1151` (Radeon 8060S, 20 WGPs, 128 GB unified
memory, 242 GB/s measured peak read). Model: `unsloth/Qwen3.8-Flash-Next-GGUF`
`UD-Q4_K_XL` (four shards, 103.7 GiB on disk, ~77 GiB resident: the 26.8 GiB
n-gram table stays on disk) with the `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`
draft block. Build: `nix develop` gpu preset, `gufo bench`.

## Throughput

`gufo bench --model ... -p 128,512,1024,2048 -b 2048 -n 64` (one untimed pass
per shape, then the timed one; single samples, the host's noise band is about
±1% with occasional 150–300 tok/s dips from background jobs):

| test | gufo | before the prefill work (`97b445b`) | llama.cpp ROCm `41abbfd59` (`-fa 1 -ub 2048`) |
| --- | ---: | ---: | ---: |
| pp128 | 464 | 443 | — |
| pp512 | 796 | 628 | 402 |
| pp1024 | 921 | — | — |
| pp2048 | 1007 | 719 | 439 |
| tg64 greedy | 23.4 | 22.8 | 21.9 |
| tg128 MTP (`--speculative mtp --draft-tokens 3 --draft-vocab 65536`) | 32.9 (depth 0) / 36.5 (depth 1024) | 38.4 / 35.0 | — |

The MTP figures are acceptance-bound: the per-cycle cost is unchanged (75.5
versus 76.5 ms) and the depth-1024 prompt accepts 85/126 drafts against
83/131 before, but the depth-0 prompt's greedy continuation diverges after the
prefill numerics changed (W8A8 and F16 mixer inputs instead of the Q8 MMQ
tier) and that continuation happens to accept 76/153.

Batched prefill against the sequential reference at 2048 tokens: finite,
scalar winner rank 1, RMSE 0.30, cosine 1.00, max error 1.96 (128 tokens
before the work: RMSE 0.61, cosine 0.99). Agreement with llama.cpp on a
776-token README prompt before the work: argmax 59–62/64 of the last rows,
mean KL 0.008–0.010, max 0.15.

## Prefill experiment summary

Interleaved same-binary A/B at `-p 2048 -b 2048`, each route evaluated behind
a toggle and then either made the default or deleted; the focused CTest
targets under `tests/models/qwen38_flash_next/` (label
`qwen38_flash_next`) hold the kernel-level equivalence checks.

| route | result | mechanism and evidence |
| --- | --- | --- |
| parallel direct-read upload | retained | three reader threads per staging batch: 76.7 GiB in 63.5 s versus 128.5 s |
| vec4 hyper-connection mix | retained, +8.9% (723 → 788) | four adjacent lanes per thread; HC mix 684 → 195 ms per two passes |
| vec4 hyper-connection combine | rejected (+0.2%, inside noise) | isolated 1.92x, no end-to-end signal |
| F16 mirrors of three dense Q8 shapes | retained then superseded | +3.5% on hipBLASLt for ssm_in / HC up / shared expert; replaced by W8A8 |
| row-split DeltaNet recurrence | retained, +1.5% (815 → 828) | Qwen 27B 128x128 recurrence ported: 3.93 → 2.96 ms per layer, 88 VGPRs, no scratch; the reference kernel keeps decode and snapshot batches |
| fused WMMA causal attention | retained, +4% (827 → 863) | 27B kernel over 24 x 256 heads / 2 KV heads with the output gate and the block mask fused; 12.3 → 4.1 ms per layer |
| F16 mixer input | retained, +3.4% (866 → 895) | the combine writes the next mixer's norm as F16; no activation quantize for the mixer down |
| vec4 MoE epilogue | retained, +3.4% (859 → 888) | float4 slot reduction, a quarter of the waves; 2.56 → 1.3 ms per layer |
| parallel routed id maps | retained, +2.7% (916 → 940) | three-pass scan replaces the one-warp-per-expert helper (0.6–1.0 ms per launch); bit-identical maps |
| W8A8 int8 WMMA dense GEMM | retained, +2.7% (938 → 964) | 27B blocked kernel over the untouched Q8_0 blocks, about 30 TOPS; 64-row tiles for the 320-row mixer down |
| routed int8 WMMA expert GEMM | retained at ≥ 24 rows per expert, +2.2% (965 → 986) | 128 x 48 macro tiles per expert over 16-row padded buckets, K-quant fetch with a cached block header, in-GEMM row gather; the MMQ tier stays below 1,229 tokens where it is 5–8% faster |
| W8A8 mixer down from a fused tiled-Q8 norm | retained, +2.4% (988 → 1011) | the combine also quantizes the norm per 32-block; removes the hipBLASLt plan lottery on that shape |
| fused Q4_K expert gate/up (decode) | rejected | MTP tg128 38.6 → 33.5 despite fewer cycles |
| 40-row expert vector dispatch (decode) | rejected | vector 33.3 versus tiled 38.0 tok/s |

## Where the time goes (after the prefill work)

Two profiled pp2048 passes, 4.5 s GPU-busy: dense W8A8 GEMMs 20% (about 30
TOPS, 54% of the int8 WMMA ceiling), routed expert GEMMs 34%, hyper-connection
combine and mix 10%, DeltaNet recurrence 5%, MoE epilogue and activation
quantization 5%, attention 2%. The routed GEMMs stream 1.57 GB of expert
weights per layer at about 90 GB/s equivalent on either tier: the WMMA
kernel is memory-unit bound (MemUnitBusy 73%, LDS bank conflicts 22–28%),
not matrix-core bound.

Decode streams ~6 GB per token (dense Q8 4.5 GB including the 675 MB output
head, experts 1.5 GB): the GEMVs run at ~210 GB/s, elementwise work ~7 ms,
graph replay leaves ~1.3 µs per launch of gaps.

## TODOs

- Routed WMMA GEMM: the fetch path, not the matrix cores, bounds it (ablating
  all WMMA work leaves 5.8 ms of the 5.2 ms call). Candidates: stage the
  per-32 scales and offsets once per row tile instead of per K block (they
  are half the LDS read traffic), a token-major lane mapping so a wave load
  covers whole 128-byte lines, and a fused gate/up kernel sharing the
  activation stage with the SwiGLU and the down-input quantization in its
  epilogue.
- Fuse the SwiGLU and the down-projection activation quantization into the
  up GEMM's epilogue (about 1 ms per layer of elementwise traffic).
- The hyper-connection combine and mix are bandwidth bound on the F32
  residual stream (about 250 MB per call); a BF16 residual would halve it at
  a precision cost that needs its own validation.
- pp512 and below still take the MMQ tier for the experts; a 16-row macro
  tile variant of the routed WMMA kernel could cover the 10-row buckets.

## Where the time went (before the prefill work)

A 512-token chunk touches every expert, so it streams the full ~75 GB of Q4
expert weights (~350 ms, 44% of the chunk); the vendored llama.cpp MMQ tier runs
the gate/up GEMMs at ~214 GB/s there. Dense Q8_0 projections run at 20–46 TOPS
depending on shape. At 2048 tokens the routed GEMMs become compute bound at ~14
TOPS (mean bucket 40 rows, max ~700), and the MoE tier takes 876 ms of 2.8 s.

## Expert GEMM prototype (`tools/moe_gemm_i8_bench.hip`)

A from-scratch grouped GEMM for the routed projections: int8 WMMA
(`wmma_i32_16x16x16_iu8`, wave32 lane layout probed empirically), Q4_K blocks
laid out chunk-major per expert, per-32 scales and mins applied in a float
epilogue every k-step, activations as int8 rows with per-32 scale and sum,
one register-prefetched 256-wide k chunk per pipeline step, tokens sorted by
expert into 16/32/48/64-row tiles. Synthetic gate projection, E=512, N=640,
K=2560, versus the MMQ tier on the real model:

| rows per expert | prototype | MMQ tier | note |
| ---: | ---: | ---: | --- |
| 10 (pp512 regime) | 2.63 ms | 2.2 ms | 180 GB/s vs 214; memory unit busy 99%, L2 hit 6% |
| 20 | 3.10 ms | — | |
| 40 (pp2048 regime) | 3.92 ms | 4.8 ms | 17 TOPS |
| 64 | 4.89 ms | — | 22 TOPS |

Things that did not move the 10-row case: rocWMMA f16 fragments (7.8 TOPS at
best), weights loaded per lane into registers instead of through LDS, a
row-major versus chunk-major layout (5%), prefetch depth 2–3, 4-wave blocks,
half-chunk pipeline steps (worse: partial cache lines), forcing two blocks per
WGP. The register-spill traps: dynamic indexing of the accumulator or of the
weight words puts them in scratch; unroll the fragment and group loops fully.

Integrating it would gain at most ~100 ms per 512-token chunk (mostly on the
Q5_1 down projection, which the tier runs at 161 GB/s) and ~250 ms per
2048-token chunk: roughly 628 → 720 and 719 → 800 tok/s. The production
routed WMMA kernel above took the blocked W8A8 structure instead of this
prototype's layout; the prototype stays as the standalone harness.
