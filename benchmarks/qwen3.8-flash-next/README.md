# Qwen3.8-Flash-Next on Strix Halo

Linux x86-64, AMD Strix Halo `gfx1151` (Radeon 8060S, 20 WGPs, 128 GB unified
memory, 242 GB/s measured peak read). Model: `unsloth/Qwen3.8-Flash-Next-GGUF`
`UD-Q4_K_XL` (four shards, 103.7 GiB on disk, ~77 GiB resident: the 26.8 GiB
n-gram table stays on disk) with the `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`
draft block. Build: `nix develop` gpu preset, `gufo bench`.

## Throughput

`gufo bench --model ... -p 512,2048 -b 2048 -n 64 -r 3` (one untimed pass per
shape, then three timed):

| test | gufo | llama.cpp ROCm `41abbfd59` (`-fa 1 -ub 2048`) |
| --- | ---: | ---: |
| pp512 | 628 | 402 |
| pp2048 | 719 | 439 |
| tg64 greedy | 22.8 | 21.9 |
| tg128 MTP (`--speculative mtp --draft-tokens 3 --draft-vocab 65536`) | 38.5 | — |

Agreement with llama.cpp on a 776-token README prompt: argmax 59–62/64 of the
last rows, mean KL 0.008–0.010, max 0.15.

## Where the time goes

A 512-token chunk touches every expert, so it streams the full ~75 GB of Q4
expert weights (~350 ms, 44% of the chunk); the vendored llama.cpp MMQ tier runs
the gate/up GEMMs at ~214 GB/s there. Dense Q8_0 projections run at 20–46 TOPS
depending on shape. At 2048 tokens the routed GEMMs become compute bound at ~14
TOPS (mean bucket 40 rows, max ~700), and the MoE tier takes 876 ms of 2.8 s.

Decode streams ~6 GB per token (dense Q8 4.5 GB including the 675 MB output
head, experts 1.5 GB): the GEMVs run at ~210 GB/s, elementwise work ~7 ms,
graph replay leaves ~1.3 µs per launch of gaps.

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
2048-token chunk: roughly 628 → 720 and 719 → 800 tok/s. It is not wired into
the runtime.
