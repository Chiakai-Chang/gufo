# Qwen3.8-Flash-Next on Strix Halo

Linux x86-64, AMD Strix Halo `gfx1151` (Radeon 8060S, 20 WGPs, 128 GB unified
memory, 242 GB/s measured peak read). Model: `unsloth/Qwen3.8-Flash-Next-GGUF`
`UD-Q4_K_XL` (four shards, 103.7 GiB on disk, ~77 GiB resident: the 26.8 GiB
n-gram table stays on disk) with the `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`
draft block. Build: `nix develop` gpu preset, `gufo bench`.

## Throughput

`gufo bench --model ... -p 128,512,1024,2048 -b 2048 -n 64` (one untimed pass
per shape, then the timed one; single samples, the host's noise band is about
±1% with occasional 150–300 tok/s dips from background jobs; the GPU clock
ramps from 1 to 2.76 GHz over the first seconds of a run, so a first timed
pass can read 2–4% low — `-p 2048,2048,2048,2048` settles at 1294–1301):

| test | gufo (`nix build` of this tree) | before the F16 expert card (`72fe0f4`) | before the prefill work (`97b445b`) | llama.cpp ROCm `41abbfd59` (`-fa 1 -ub 2048`) |
| --- | ---: | ---: | ---: | ---: |
| pp128 | 474 | 464 | 443 | — |
| pp512 | 814 | 796 | 628 | 402 |
| pp1024 | 1195 | 931 | — | — |
| pp2048 | 1293 (steady clock 1298) | 1040 | 719 | 439 |
| tg64 greedy | 22.9 | 22.7 | 22.8 | 21.9 |
| tg128 MTP (`--speculative mtp --draft-tokens 3 --draft-vocab 65536`) | 29.6 (depth 0) / 36.2 (depth 1024) / 14.8 (depth 4096) / 15.3 (depth 16384) | 32.9 (depth 0) / 36.5 (depth 1024) / 35.0 (depth 4096) / 42.9 (depth 16384) | 38.4 / 35.0 / 30.4 / — | — |

Context depth (`-p 2048 -n 64 -b 2048 -d N`, one 2048-token chunk and 64
decode steps after N prepared tokens; the sparse attention budget is 2048
tokens, so past 2048 keys every full-attention layer selects blocks):

| depth | pp2048 | tg64 | pp2048 before the selection card | tg64 before | pp2048 before the F16 expert card | tg64 before | pp2048 before the depth card | tg64 before |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1276 | 22.2 | 1299 | 22.5 | 1016 | 22.8 | 1010 | 22.8 |
| 4096 | 1210 | 22.0 | 1172 | 21.7 | 942 | 21.7 | 484 | 17.8 |
| 16384 | 1155 | 21.5 | 1051 | 20.9 | 838 | 20.1 | 320 | 15.0 |
| 32768 | 1109 | 20.9 | 936 | 19.5 | 766 | 19.4 | 221 | 12.4 |
| 65536 | — | — | — | — | — | — | 140 | — |

Where a 2048-token chunk's time goes with depth (one profiled pass per depth,
12 full-attention layers; everything else is flat at about 1,560 ms):

| depth | attention | block scoring (indexer) | top-k marking |
| ---: | ---: | ---: | ---: |
| 0 | 46 ms | 0 | 0 |
| 2048 | 162 | 2 (was 25) | 7 (was 10) |
| 8192 | 219 | 5 (was 64) | 8 (was 22) |
| 16384 | 248 | 8 (was 128) | 11 (was 44) |

The attention kernel's growth is the gather of the four-query union of
selected blocks (about 850 blocks, 3,400 keys, 7 MB of K/V per block of four
queries at 16k: 3.6 GB per layer at roughly 175 GB/s), i.e. K/V bandwidth,
not matrix work; a narrower K/V format is the lever left there.

Depths of 64k and beyond were not benchmarked after the change (each depth
prepares the whole context first); the per-chunk cost is now the union of
four consecutive queries' selections (about 850 blocks at 16k against 512
per query) plus a per-query indexer scan, so it still grows slowly with
depth. Peak resident memory at the 65k depth measured 90.9 GiB GTT (77 GiB
weights; KV is 24 KiB per token).

The MTP figures are acceptance-bound on this synthetic prompt: the per-cycle
cost is unchanged (76.3 ms now, 75.9 before the F16 expert card, 75.5 / 76.5
across the earlier cards), depth 1024 accepts 82/133 drafts (85/126 before),
and depth 0 accepts 71/168 on a greedy continuation that diverged when the
prefill numerics changed. At depths 4096 and 16384 the current greedy
continuation of the repeated benchmark sentence emits `<|im_end|>` at once
and the draft cannot follow what comes after it (21/315 and 29/291
accepted); the binary before the card continued the sentence there (83/132
at 4096). Forcing any one of the wide MoE passes onto the MMQ tier flips the
continuation back (77/147), so this is the same numerics-sensitive fork as
at depth 0, not a draft-path defect: the greedy tg64 continuation at depth
4096 is identical between the two binaries.

Batched prefill against the sequential reference at 2048 tokens: finite,
scalar winner rank 1, RMSE 0.26, cosine 1.00, max error 1.73 (RMSE 0.30, max
error 1.96 before the F16 expert card; 128 tokens before the prefill work:
RMSE 0.61, cosine 0.99). Agreement with llama.cpp on a
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
| routed int8 WMMA expert GEMM | retained at ≥ 24 rows per expert, +2.2% (965 → 986) | 128 x 48 macro tiles per expert over 16-row padded buckets, K-quant fetch with a cached block header, in-GEMM row gather; the MMQ tier stays below 820 tokens where it is 5% faster |
| W8A8 mixer down from a fused tiled-Q8 norm | retained, +2.4% (988 → 1011) | the combine also quantizes the norm per 32-block; removes the hipBLASLt plan lottery on that shape |
| sparse-window attention at depth (depth card) | retained, pp2048@16k 320 → 838, tg64@32k 12.4 → 19.4 | the fused WMMA kernel was only used up to 4096 keys and the per-token fallback swept every key of the context with the mask; now: the WMMA kernel at any depth gathers 16-key tiles from the union of a block's query masks, with the twelve heads of one query packed into a 16-row tile and four queries per block (a 32-query union covered 60% of the context, a 4-query union 20%); the per-token kernel compacts the query's selected blocks through LDS, splits its tiles over eight blocks with a log-sum-exp merge, and scores/accumulates wave-cooperatively (one 16-byte load per lane) instead of one lane per key; block selection scores sixteen queries per pooled key row over row splits instead of one block per query |
| SwiGLU in the routed up projection's epilogue | retained (bundled) | the up GEMM writes silu(gate) * up in place of the up result; drops the SwiGLU pass and one 52 MB round trip per layer |
| routed WMMA route from 16 rows per expert (was 24) | retained, pp1024 914 → 931 | after the LDS padding the WMMA tier wins at 1024 tokens too; pp512 (ten-row buckets) stays on MMQ |
| routed GEMM LDS plane padding | retained, +0.7% (1014 → 1022) | one-element padding per K-block plane halves the bank conflicts (22–28% → 11–18% of cycles); MemUnitBusy then reads 93–95%, so the kernel is memory-unit bound at ~15 TOPS |
| routed GEMM tiles 128x96 (4x2 waves) / BK=2 | rejected | 6.53 / 8.51 ms and 4.76 / 5.77 ms per call against 4.45 / 5.36 for 128x48 BK=4: fewer weight re-fetches lose to the occupancy drop (256 VGPRs, 34 KB LDS) |
| routed F16 WMMA expert GEMM (replaces the int8 kernel) | retained, pp2048 1040 → 1162 | the int8 kernel paid a float epilogue and an activation-sum correction every K block (VALU and WMMA serialize on gfx1151: WMMA 1.63 ms + VALU 0.66 ms of a 4.3 ms Q4_K call); the F16 kernel dequantizes the codes after the LDS read (packed nibbles / raw Q5_1 blocks in LDS, 11–12 KB, five blocks per WGP) with a byte permute into the mantissa of 1024.0, an exact packed subtract and one packed FMA, reads F16 activation rows (the router's narrow of x; the up projection's epilogue writes the down projection's F16 input) and accumulates all of K in F32: Q4_K gate/up 4.3 → 3.3 ms, Q5_1 down 5.3 → 4.4 ms in the harness (10x closer to F64 than the int8 route: 1.33 vs 12.2 at scale 4036); a host-built tile map (expert \| tile << 16) launches no empty blocks and the grid runs row blocks fastest so a tile's gathered activations stay in L2 (tiles-fastest was 4.5 / 9.4 ms) |
| Q8_0 and Q5_K experts on the F16 route | retained, +55 tok/s with the combine below | the five Q8_0-down layers ran the MMQ tier at 12 ms per down call (4.4 ms on the F16 route: signed bytes carried as q + 128 with a 1152 magic) and the Q5_K layer's tier calls sat behind 7.8 ms host gaps between its quantize and GEMM launches; every wide MoE layer now takes the F16 route |
| F16 down rows + F16 MoE epilogue | retained, +5 (bench) / −22 ms per pass (profile) | the down projection writes F16 rows and the epilogue reads them: half the bytes of the largest routed intermediate |
| one block per token hyper-connection combine | retained, 1.71 → 1.27 ms per call | float4 lanes over all four streams of a token, the updated residual kept in registers between the two passes, one block reduction for the four norms, four-code stores into the tiled Q8 norm |
| multi-token hyper-connection mix | retained, 0.93 → 0.83 ms per call | eight tokens per block reuse the block's inject weight slice from registers (sixteen weight loads per token dominated the one-token kernel) |
| sliding-window causal conv | retained, 1.27 → 0.80 ms per call | one thread per (channel, eight tokens): the four taps in one float4, the window in registers, one load and one store per token |
| vec4 tiled Q8 quantizer | retained, ≈ −20 ms per pass | four elements per lane, eight lanes per block, four-code stores |
| activation-pass fusions | retained, ≈ −35 ms per pass | the GDN epilogue writes the ssm_out projection's tiled Q8 input, the mix writes the F16 and tiled Q8 copies of `mixed` that the router / routed GEMMs and the mixer in-projection read, the shared expert's SwiGLU writes its down projection's tiled input, and an input cache lets a projection over rows that are already staged skip its pass; the shared expert is launched before the expert-count readback so the device stays busy through it |
| MoE epilogue fused into the combine | retained, 2.05 → 1.75 ms per layer | the block output row is formed in registers from the experts' F16 rows and never written |
| BM=256 (two row tiles per wave) F16 expert GEMM | rejected | 4.46 / 4.40 ms: spills at m = 640 (three row blocks, the last half empty) and no gain for the down projection, so the activation LDS reads are not the bound |
| two-stage weight prefetch, 128-byte-per-row fetch rounds | rejected | 4.4 / 4.4 and 4.7–4.9 / 4.6–5.1 ms: the compiler caps VGPRs at 144–152 for the LDS-derived occupancy and spills; ablations put the F16 kernel at 2.3 / 2.9 ms with no weight loads, 2.7 / 3.1 with fully coalesced ones and 3.2 / 3.7 with the real pattern served from cache, so the remaining cost is the sixteen-lines-per-instruction fetch itself (about 0.5 ms) plus 0.1 / 0.7 ms of exposed DRAM |
| 64x64 tiles for the 320-row mixer down | rejected | 0.47 vs 0.40 ms per call: the weight re-reads cost more than the occupancy gained |
| fused gate+up F16 expert GEMM (one launch, shared activation stages, SwiGLU on the accumulators) | rejected | 7.1 ms against 2 x 3.3: two accumulator sets and two decoded row tiles put the kernel at 231 VGPRs with 48 bytes of scratch (indexing the block header by a runtime byte lands it in memory) and 24.7 KB of LDS after the compiler promotes an alloca, two blocks per WGP |
| F16 gate rows (the up epilogue reads the gate as F16) | rejected | neutral in a four-repeat interleaved A/B (1287–1298 against 1294–1301); the gate round trip is 0.2 ms per layer of a 1.6 s pass |
| indexer block scoring on the matrix cores (selection card) | retained, pp2048@16k 1051 → 1155, @32k 936 → 1109 | the scalar scan (four heads x 128 dims per query-block pair, 8.6 GFLOP per layer at 16k) ran at 0.8 TFLOPS; now 32 queries x 128 blocks per workgroup as F16 fragments read straight from the F32 rows, four per-head F32 accumulators, relu-sum in the epilogue: 128 → 8 ms per chunk at 16k (scores within 0.6% of F64; the mask over them is exact). The unrolled K loop spilled 1 KB per lane and read garbage; `#pragma unroll 1` holds it at 147 VGPRs |
| top-k marking by four-pass radix select | retained (bundled) | the 32-pass bit search re-read every score from L2 per pass; a 256-bin histogram per byte finds the threshold in four passes and a ballot scan ranks the ties in index order: 44 → 11 ms per chunk at 16k; decode at depth gains too (tg64@32k 19.5 → 20.9) |
| fused Q4_K expert gate/up (decode) | rejected | MTP tg128 38.6 → 33.5 despite fewer cycles |
| 40-row expert vector dispatch (decode) | rejected | vector 33.3 versus tiled 38.0 tok/s |

## Where the time goes (after the F16 expert card)

One profiled pp2048 pass, 1.60 s GPU-busy (idle between dispatches 7 ms):
dense W8A8 GEMMs 30% (about 32 TOPS, 58% of the WMMA ceiling; the 16384-row
GDN in-projection alone is 12%), routed expert GEMMs 34% (Q4_K gate/up 3.5
ms, Q5_1 down 4.5 ms per layer), DeltaNet recurrence 7%, hyper-connection
combine (with the MoE epilogue) and mix 15%, attention 3%, causal conv 2%.
VALU and WMMA do not overlap on gfx1151, so the expert GEMM's floor is its
WMMA issue (1.63 ms per Q4_K call) plus its dequant VALU (0.66 ms); the
measured 3.3 ms adds the weight fetch (sixteen cache lines per wave-wide
load, about 0.5 ms) and exposed DRAM latency. The int8 dense kernel's float
epilogue is 17% of its time (ablation).

Before the card (two passes, 4.5 s): dense W8A8 GEMMs 20%, routed expert
GEMMs 34%, hyper-connection combine and mix 10%, DeltaNet recurrence 5%,
MoE epilogue and activation quantization 5%, attention 2%; the int8 routed
kernel streamed 1.57 GB of expert weights per layer at about 90 GB/s
equivalent (MemUnitBusy 73%, LDS bank conflicts 22–28%).

Decode streams ~6 GB per token (dense Q8 4.5 GB including the 675 MB output
head, experts 1.5 GB): the GEMVs run at ~210 GB/s, elementwise work ~7 ms,
graph replay leaves ~1.3 µs per launch of gaps.

## TODOs

- 1,400 tok/s at 2048 tokens needs about 115 ms less per pass; the two
  structural cards below are the only levers of that size left (the
  elementwise fusions above took the rest).
- Routed F16 GEMM: the weight fetch reads sixteen 32-byte row pieces per
  wave-wide load; 128 contiguous bytes per row per instruction needs four
  stages of codes in registers, which the compiler spills at the
  occupancy it targets (rejected rounds above). A fragment-major repack of
  the expert codes at load time would make each stage's sixteen rows one
  512-byte read, but the decode GEMVs read the standard layout, so it needs
  either a second copy (72 GB) or repacked decode kernels. A fused gate+up
  kernel sharing the activation stage halves the activation traffic and the
  gate round trip.
- Dense W8A8 GEMM at 58% of the WMMA ceiling: the per-K-block float
  epilogue is 17% of its time; an F16 formulation like the expert GEMM
  (Q8_0 codes dequantized after the LDS read, F16 activations from the
  producers) removes it and the activation passes, but every dense input
  would have to be produced as F16.
- DeltaNet recurrence (3.0 ms per layer, 7% of a pass) is a sequential
  per-token chain; a chunked formulation would turn it into GEMMs.
- Attention at depth is bound by the gather of the four-query union of
  selected blocks (3.6 GB of K/V per layer at 16k); a narrower K/V cache
  (FP8 with per-block scales) would halve it at a precision cost that needs
  validation, and a 5-query x 12-head flat row layout (60 live rows of 64)
  would cut the sweep another ~10%.
- The draft block folds its four streams into rows, so its projections run
  at 4x the batch; the W8A8 path chunks them to the tiled buffer (found as
  a GPU fault in MTP decode past 2048 tokens of context: the first card's
  W8A8 route overflowed that buffer silently below 4096 and faulted above).
- The hyper-connection combine and mix are bandwidth bound on the F32
  residual stream (about 250 MB per call); a BF16 residual would halve it at
  a precision cost that needs its own validation.
- pp512 and below still take the MMQ tier for the experts; a 16- or 32-row
  token tile variant of the routed F16 kernel could cover the 10-row
  buckets (the tile map already handles ragged buckets).

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
