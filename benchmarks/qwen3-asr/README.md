# Benchmark: Qwen3-ASR-1.7B transcription on Strix Halo

Status: retained, 2026-08-30. Native C++/HIP route on `gfx1151`, BF16 weights,
deterministic greedy decode. Baseline is revision `9686079`.

Methodology rules in [../../docs/PERFORMANCE.md](../../docs/PERFORMANCE.md).
Headline latency comes from unprofiled runs; kernel times come from separate
`rocprofv3` runs because tracing changes timing. Every route comparison below
was collected by alternating routes inside one process on one thermal state --
the APU's clocks drift enough across sessions that cross-session comparisons are
not valid.

## Fixture and method

- Audio: the upstream English fixture, `15.051229 s`, 48 kHz mono PCM24.
- Prompt: 211 text tokens, of which 196 are audio embedding positions.
- Generation: 49 tokens, greedy, ending at `151645`.
- Command: `src/models/qwen3_asr/tools/benchmark.py`, one resident runtime,
  one warmup, five measured runs, medians reported.
- Model: `Qwen/Qwen3-ASR-1.7B` snapshot
  `7278e1e70fe206f11671096ffdd38061171dd6e5`.

Two rates are reported. `real-time factor` is wall time divided by audio
duration, so lower is better and anything below `1.0` is faster than real time.
`audio/wall` is its reciprocal: seconds of audio transcribed per second of wall
clock, which is the "N times faster than real time" figure.

## Headline

| | baseline `9686079` | retained | change |
|---|---:|---:|---:|
| Total request | `2046.78 ms` | `992.17 ms` | **`2.06x` faster** |
| Real-time factor | `0.13599` | `0.06592` | `2.06x` lower |
| Audio per wall second | `7.35x` | **`15.17x`** | `+7.82x` real time |
| Weight load | `0.62 s` | `0.60 s` | unchanged |
| GPU kernel time | `1745.67 ms` | `963.59 ms` | `1.81x` less |
| GPU dispatches | `27144` | `13755` | `1.97x` fewer |

The retained route transcribes 15 seconds of speech in under a second, and it
is `5.25x` faster than the official BF16 implementation's `5.20 s` generate-only
time on the same fixture -- while also doing WAV decoding, resampling, and
feature extraction, which that figure excludes.

Every generated token ID and the final transcript are unchanged. See
[Quality](#quality).

## Stage breakdown

| Stage | baseline | retained | change |
|---|---:|---:|---:|
| WAV decode + resample to 16 kHz (CPU) | `162.63 ms` | `6.27 ms` | `25.9x` |
| Log-mel feature extraction (CPU) | `83.93 ms` | `7.18 ms` | `11.7x` |
| Audio encoder (GPU) | `51.04 ms` | `36.02 ms` | `1.42x` |
| Text decoder: prefill + 48 decode steps (GPU) | `1748.65 ms` | `942.44 ms` | `1.86x` |
| Total | `2046.78 ms` | `992.17 ms` | `2.06x` |

Both columns are single benchmark invocations so the stages sum to the total.
Three consecutive invocations of the retained route landed between `991.8 ms`
and `992.9 ms`, a spread of `0.11%`.

## Route comparison

All routes in one process, two alternating passes, medians. Each reproduces the
same 49 official token IDs and the same transcript.

| Route | Total | Text decoder | Real-time factor | Audio / wall |
|---|---:|---:|---:|---:|
| Retained | `990.0 ms` | `934.5 ms` | `0.0658` | `15.20x` |
| `GUFO_QWEN3_ASR_TEXT_ATTENTION=batched` | `1028.7 ms` | `973.8 ms` | `0.0684` | `14.63x` |
| `GUFO_QWEN3_ASR_PREFILL_GEMM=hipblas` | `1079.2 ms` | `1014.1 ms` | `0.0717` | `13.95x` |
| `GUFO_QWEN3_ASR_TEXT_GEMV=unfused` | `1673.3 ms` | `1618.8 ms` | `0.1111` | `9.00x` |
| All three reverted together | `1808.2 ms` | `1742.9 ms` | `0.1201` | `8.32x` |
| `GUFO_QWEN3_ASR_TEXT_GEMV=hipblas` | `2472.3 ms` | `2417.2 ms` | `0.1642` | `6.09x` |

`unfused` is the previously retained batch-one route: one shared
`Wave32GEMVKernel_1Row` dispatch per projection. Reverting all three GPU routes
lands at `1808 ms` rather than the `2047 ms` baseline because the two CPU audio
fixes are not switchable.

## GPU kernel breakdown

`rocprofv3`, one unwarmed request each.

| Kernel | baseline | retained |
|---|---:|---:|
| Decode projections | `1385.57 ms` / 9457 calls | `747.58 ms` / 2786 calls |
| Decode attention | `88.70 ms` / 1344 calls | `50.13 ms` / 1344 calls |
| Prefill and audio-tower GEMM | `187.56 ms` / 391 calls | `106.63 ms` / 391 calls |
| Residual-add + RMSNorm | `24.45 ms` / 2688 calls | `24.64 ms` / 2688 calls |
| Audio-tower self-attention | `10.31 ms` / 24 calls | `7.37 ms` / 24 calls |
| Prefill attention | `7.96 ms` / 28 calls | `6.13 ms` / 28 calls |
| Argmax over 151936 logits | `6.50 ms` / 49 calls | `0.28 ms` / 147 calls |
| BF16-to-float32 activation copies | `9.42 ms` / 9457 calls | removed |
| Total kernel time | `1745.67 ms` | `963.59 ms` |

Decode projections are now `77.8%` of GPU time and sit at the memory roofline,
so they are no longer the productive target. See below.

## What changed

### 1. Fused, LDS-staged batch-one decode GEMV (`-683 ms`)

Batch-one decode reads all 3.441 GB of decoder weights per token -- 28 layers of
q/k/v/o/gate/up/down plus the 622 MB language-model head -- so it can only be as
fast as `gfx1151` streams memory. The measured DRAM read ceiling is
`240.24 GB/s` (`tools/bench/gfx1151_peak.hip`); the previous route sustained
`123.9 GB/s`, half of it.

`tools/bench/asr_decode_gemv_bench.hip` allocates the whole per-token weight
arena and times a complete token pass. That structure matters: a per-shape
microbenchmark keeps a 4-25 MB projection resident in the 32 MB MALL and reports
400-860 GB/s for a kernel that sustains 124 GB/s against cold memory.

| Variant | ms/token | GB/s | of peak | numerics |
|---|---:|---:|---:|---|
| Previous route: 1 row/wave, float32 activations | `27.78` | `123.9` | `51.6%` | reference |
| BF16 activations | `27.71` | `124.2` | `51.7%` | bit-exact |
| + 8-deep row prefetch | `20.28` | `169.7` | `70.6%` | bit-exact |
| + q/k/v and gate/up fused into one dispatch | `18.40` | `187.0` | `77.8%` | bit-exact |
| + activations staged in LDS (**retained**) | `14.78` | **`232.8`** | **`96.9%`** | bit-exact |
| Streaming-read ceiling, same geometry | `14.83` | `232.0` | `96.6%` | n/a |

Three mechanisms, each measured separately:

- **Activation cache contention was the largest single cost.** A ceiling kernel
  that streams only weights runs at `232.0 GB/s`; adding the activation load
  stream and nothing else drops it to `177.2 GB/s`. Staging the activation
  vector in LDS once per block removes that contention entirely and is what
  takes the kernel from 78% to 97% of peak.
- **Loads in flight.** One 16-byte row load per lane per iteration leaves the
  memory pipeline short; eight is the measured optimum. Sixteen regresses to
  `159.3 GB/s` on register pressure.
- **Occupancy on narrow projections.** One wave per row gives k_proj and v_proj
  (M=1024) only 6.4 waves per SIMD against a 16-wave limit. q/k/v share one
  activation vector, as do gate/up, so each runs as a single dispatch over
  concatenated rows: 4096 and 12288 rows instead of three and two smaller grids.

The retained kernel matches the streaming-read ceiling, so the decode
projections are now bandwidth-bound with no headroom left. Reading fewer bytes
would mean quantizing the weights, which this benchmark's accuracy contract
rules out.

Non-temporal loads were tried and rejected: `__builtin_nontemporal_load` bypasses
the MALL and costs more than half the throughput (`177.5 GB/s` at unroll 4,
against `232.8` retained). Two and four rows per wave were also rejected
(`170.6` and `120.7 GB/s`) -- they trade the wave count the machine needs for
activation reuse that LDS already provides for free.

### 2. hipBLASLt for the multi-token projections (`-89 ms`)

The audio tower and the text prefill multiply a transposed BF16 weight by BF16
activations into a float32 result. rocBLAS has no WMMA kernel for that
combination and resolves it to a scalar `MT64x32x8` Tensile kernel at `3.0-4.8
TFLOPS` against a `55 TFLOPS` WMMA ceiling. hipBLASLt is `1.5-5.2x` faster on
every one of these shapes, so both runtimes now try hipBLASLt first and keep
hipBLAS as the fallback.

This is the one retained change that alters numerics, and it improves them: see
[Quality](#quality).

### 3. Model-private decode attention (`-39 ms`)

The shared batched kernel issues a single 16-byte key load per wave per position
and reduces it immediately, which leaves it bound by load latency at roughly an
eighth of the DRAM read rate -- and with one block per head, 16 of 40 compute
units. The model-private kernel keeps every arithmetic step and only raises
loads in flight: each wave scores 8 positions at a time and the value
accumulation prefetches 8 rows before consuming the first. `88.70 ms` to
`50.13 ms`, `1.77x`, bit-identical.

Per-position score reductions are independent, and the context accumulator stays
single and in ascending position order, so the result cannot change. A context
shorter than 64 positions falls back to the shared kernel, which switches to a
different accumulation order below that length.

### 4. Resampler phase memoization (`-155 ms`)

The fixture is 48 kHz, so the windowed-sinc resampler runs 48 taps per output
sample and was calling `sin` and `cos` inside every tap -- 23 million
transcendental evaluations for 15 seconds of audio.

Each tap weight depends only on `position - source`, and with
`source = floor(position) + tap` that is the correctly-rounded value of
`frac - tap`. Two output frames with a bit-identical `frac` therefore produce
bit-identical weights, so the weights collapse to one evaluation per distinct
fractional phase: a single phase for any exact multiple of 16 kHz, 160 phases
for 44.1 kHz. `162.63 ms` to `6.24 ms`.

### 5. Threaded log-mel front end (`-77 ms`)

The windowed DFT is 201 bins times 400 samples per frame, about a hundred frames
per audio second, single threaded. Frames are independent and each keeps its own
accumulation order, as do mel bands, so both loops now spread across cores.
`83.93 ms` to `7.11 ms`. The frontend output is bit-identical -- its quality row
is unchanged to every printed digit.

### 6. Audio-tower attention operand loads (`-3 ms`)

The audio tower scores one block per (query token, head) with 64 threads, and
the score loop read both operands one float at a time -- 79 million scalar loads
for 15 seconds of audio -- while re-reading the block's invariant query row once
per key. The query row now lives in registers and both operands load 16 bytes at
a time; the value accumulation prefetches 8 rows. Every fmaf chain keeps its
ascending order and single accumulator, so the result is bit-identical: the
encoder's three quality rows are unchanged to every printed digit.
`10.31 ms` to `7.64 ms`, `1.35x`. End to end this is inside run-to-run noise.

### 7. Grid-wide argmax (`-6 ms`)

Selecting a token scanned all 151936 logits in a single block, on one compute
unit, at `133 us` per step. A maximum and a minimum index are both order
independent, so the scan now runs across the grid in three dispatches:
`6.50 ms` to `0.28 ms`, `23x`. End to end this is inside run-to-run noise.

## Quality

The contract is the official BF16 reference: every generated token ID, the exact
transcript, and the per-boundary comparisons below. Both columns come from the
same binary, selecting routes with the environment knobs, so they are directly
comparable.

| Boundary | Cosine before | Cosine after | Rel L2 before | Rel L2 after | Max abs before | Max abs after |
|---|---:|---:|---:|---:|---:|---:|
| CPU log-mel frontend | `1.0` | `1.0` | `9.42924e-7` | `9.42924e-7` | `7.21216e-5` | `7.21216e-5` |
| HIP convolutional frontend | `1.0` | `1.0` | `3.09794e-4` | `1.11245e-4` | `0.03125` | `0.015625` |
| Audio encoder layer 0 | `0.999998` | `0.999998` | `0.00199992` | `0.00198788` | `0.0625` | `0.0625` |
| Complete audio encoder | `0.999870` | `0.999855` | `0.0161537` | `0.0170533` | `0.00756836` | `0.00830078` |
| Text decoder layer 0 | `0.999997` | `0.999998` | `0.00234885` | `0.00194198` | `0.0625` | `0.0625` |
| Prefill logits | `0.999692` | `0.999721` | `0.0259246` | `0.023642` | `0.671875` | `0.5625` |

- Greedy generation reproduces all 49 official token IDs and the exact
  transcript on both routes.
- Prefill logits, the boundary that actually selects tokens, improved on all
  three measures. First-token argmax is `11528` on both, matching the official
  value.
- Five of six boundaries are unchanged or better. The complete audio encoder is
  marginally worse -- relative L2 `0.01615` to `0.01705` -- because hipBLASLt
  accumulates the tower's projections in a different order. Its gate allows
  `0.1`, so this sits 5.9x inside the limit, and it does not propagate: the
  text-decoder and prefill boundaries downstream of it both improved.
- The decode GEMV, decode attention, argmax, resampler, and log-mel changes are
  bit-exact by construction; only the hipBLASLt prefill route moves any number
  in this table.

## Reproduce

```sh
git add -A && nix build

nix develop -c python3 src/models/qwen3_asr/tools/benchmark.py \
  --model /var/llms/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --audio /tmp/qwen3-asr-en.wav

# Numerics and token-exactness gates.
nix develop -c ctest --preset gpu-full -R qwen3_asr --output-on-failure

# GEMV roofline study.
nix develop -c tools/bench/build.sh asr_decode_gemv_bench \
  asr_prefill_gemm_bench gfx1151_peak
/tmp/gfx1151_peak
/tmp/asr_decode_gemv_bench
/tmp/asr_prefill_gemm_bench --algos   # per-algorithm hipBLASLt table

# Kernel timing.
nix develop -c python3 tools/prof/prof.py run -o /tmp/prof/asr --stages qwen -- \
  ./result/bin/gufo transcribe --model <snapshot> --audio /tmp/qwen3-asr-en.wav \
  --warmup 0 --repeat 1 --format json
```

Route knobs, all defaulting to the retained route:

```sh
GUFO_QWEN3_ASR_TEXT_GEMV=unfused|hipblas       # batch-one decode projections
GUFO_QWEN3_ASR_TEXT_ATTENTION=batched|hipblas  # decode attention
GUFO_QWEN3_ASR_PREFILL_GEMM=hipblas            # multi-token projections
GUFO_QWEN3_ASR_WEIGHT_MODE=mapped|copy         # weight residency
```

Performance numbers must come from `result/bin/gufo`. The `build/gpu-test` tree
is `CMAKE_BUILD_TYPE=Debug` and is for correctness only -- the retained kernels
are measurably *slower* there than the routes they replace, because the unrolled
loops spill without optimization.

## Module independence

`src/models/qwen3_asr` builds with no dependency on any other model's sources,
and `gufo_qwen3_asr` no longer links `gufo_core`. Five things previously came
from `src/models/qwen` or `src/models/qwen3_tts`; each is now model-private:

| Was | Now | Note |
|---|---|---|
| `gufo::hip::LaunchGEMV` + `QwenGemmMode` | `LaunchTextRowGemv` | BF16 only; the generic version carried quantized and float32 routes this checkpoint never uses |
| `gufo::hip::LaunchBatchedEmbeddingLookup` | `LaunchTextEmbeddingLookup` | BF16 only |
| `gufo::hip::LaunchBatchedAttention` | `LaunchTextBatchedAttention` | float32 cache, no attention gate, no KV write -- the only route this model takes |
| `gufo::hip::LaunchHipblasGEMMBF16`, `HipblasLtGemm` | `LaunchGemmBf16`, `GemmLt` | no plan database, no tuning; selection policy is this model's own |
| `qwen3_tts::json`, `qwen::QwenTokenizer` | `qwen3_asr::json`, `BpeTokenizer` | JSON copied verbatim; the tokenizer keeps its encode/decode algorithm unchanged and drops the GGUF and binary-vocabulary constructors, which is what removed the last `src/core` dependency |

The copies are tailored rather than verbatim where the original was mostly
generality this model never reaches -- a verbatim copy of 755 lines of hipBLASLt
plan-database machinery that is never called would be worse for the codebase
than a 220-line wrapper that does what this model needs.

Verifying a refactor of numerics-carrying kernels needs a bisect, not a build:
with `GUFO_QWEN3_ASR_PREFILL_GEMM=hipblas` the copied embedding, attention, and
hipBLAS paths reproduce the pre-refactor quality metrics to every printed digit,
which isolates all remaining movement to hipBLASLt algorithm selection.

Two measured side effects:

- The model-private prefill attention is `1.32x` faster than the shared kernel it
  replaced (`7.96 ms` to `6.13 ms`) for the same arithmetic, because dropping the
  unused f16-cache and gate branches lets the compiler hoist the cache addressing
  out of the key loop.
- Owning `GemmLt` made the algorithm-selection fix below possible without
  touching any shared code.

## Prefill GEMM: what is actually left

`tools/bench/asr_prefill_gemm_bench.hip` scores every engine at the nine
distinct prefill and audio-tower shapes. It streams cold weight replicas, which
matters as much here as it does for the decode GEMV: measured warm, algorithm #4
looks like the winner for `gate/up`, and measured cold it is the worst of the top
five.

Summed over one instance of every shape:

| Engine | Time | Note |
|---|---:|---|
| hipBLAS (rocBLAS/Tensile) | `5.70 ms` | no WMMA kernel for this operand combination |
| hipBLASLt, production algorithm pick | `3.19 ms` | retained |
| hipBLASLt, fastest algorithm per shape | `2.17 ms` | **`1.47x` available** |
| Repository blocked BF16 WMMA kernel | `5.44 ms` | correct, `2.5x` slower than hipBLASLt |

Two conclusions:

- **A hand-written WMMA kernel is not the win.** The repository's blocked BF16
  WMMA GEMM (`tools/bench/bf16_gemm_bench.hip`, four tile shapes tried) is
  correct at these shapes but `2.5x` slower than hipBLASLt's best algorithm, and
  loses to plain hipBLAS on four of nine shapes. At batch 196-211 it reaches
  `15-28 GB/s` of weight bandwidth against hipBLASLt's `44-86 GB/s`. Beating a
  tuned library here would be a substantial kernel project with an uncertain
  outcome, so it was rejected on measurement rather than attempted.
- **Algorithm selection is the win, and it is mis-selected today.**
  `SelectHipblasLtHeuristicRank` in
  `src/models/qwen/hip/kernels/hipblaslt_gemm.hip` is a rank policy calibrated on
  Qwen3.8-27B's shapes at batch 2048. On ASR it fires `kTallRank` for `gate/up`
  and `down` -- the two largest -- where those are the two shapes that most want a
  high-index algorithm:

| Shape | Production pick | Fastest | Gap |
|---|---|---|---:|
| text gate/up 6144x2048 | algo #4, `1.017 ms` | algo #8, `0.510 ms` | `2.00x` |
| text down 2048x6144 | algo #4, `1.015 ms` | algo #8, `0.568 ms` | `1.79x` |
| audio fc1 4096x1024 | algo #4, `0.207 ms` | algo #8, `0.163 ms` | `1.27x` |
| audio qkv/o 1024x1024 | algo #8, `0.060 ms` | algo #4, `0.050 ms` | `1.20x` |
| the other five | -- | -- | `1.00-1.08x` |

hipBLASLt's own top heuristic pick (#0) is never best. Now that `GemmLt` is
model-private the rank is chosen by an end-to-end sweep over this model's shapes
rather than inherited:

| Uniform rank | Total request | Prefill cosine | Prefill rel L2 |
|---|---:|---:|---:|
| 0 | `1057.5 ms` | `0.999600` | `0.02830` |
| 2 | `1088.9 ms` | `0.999662` | `0.02670` |
| 4 | `1020.2 ms` | `0.999623` | `0.02806` |
| **6 (retained)** | `992.2 ms` | **`0.999703`** | **`0.02444`** |
| 7 | `982.2 ms` | `0.999671` | `0.02682` |
| 8 | `981.0 ms` | `0.999671` | `0.02682` |
| inherited qwen policy (mixed 8/4/4/4) | `1013.9 ms` | `0.999482` | `0.03226` |
| hipBLAS, no hipBLASLt | `1076.5 ms` | `0.999692` | `0.02592` |

Rank 6 is retained rather than the fastest. Rank 8 is `11 ms` (`1.1%`) better,
but rank 6 is the only rank at least as accurate as the reference route on every
prefill boundary, and this model's contract is exact reproduction of the official
token IDs. Set `GUFO_QWEN3_ASR_GEMM_RANK=8` to take the speed instead; every rank
in the table reproduces all 49 official token IDs and the exact transcript.

The inherited policy is the worst row bar the library defaults -- slower *and*
less accurate than any uniform choice -- which is what makes a foreign
calibration worth removing rather than porting.

A rank is an index into a library-generated list, so it is only valid for the
ROCm and hipBLASLt version it was measured against; re-sweep after a toolchain
bump. The remaining `1.47x` in the table above needs measured, persisted plans,
and the repository's tuner has a prerequisite bug:

> `HipblasLtGemm::TuneBf16` times every candidate against a single reused weight
> buffer, so it measures MALL-warm. For any projection smaller than the 32 MB
> MALL it would install the warm winner, which for `gate/up` is the algorithm
> that is `2.00x` too slow cold. Qwen3.8-27B never exposed this because its
> projections are 178 MB.

## Attention: no FlashAttention, and it would not pay

None of the three attention paths uses FlashAttention or the matrix cores. All
three are hand-written VALU kernels:

| Path | Kernel | Shape | Cost |
|---|---|---|---:|
| Audio tower, bidirectional | `SegmentedSelfAttentionKernel` | 196 tokens, 16 heads, head dim 64 | `7.64 ms` |
| Text prefill, causal | `BatchedAttentionKernel` | 211 tokens, 16 heads / 8 KV, head dim 128 | `7.96 ms` |
| Text decode, causal | `TextDecodeAttentionKernel` | 1 query, context to 260 | `50.13 ms` |

The repository does carry WMMA and flash-attention machinery -- Composable
Kernel's `device_grouped_query_attention_forward_wmma` and AOTriton's `attn_fwd`
-- but both are gated in `src/models/qwen/hip/detail/attention_policy.hpp` to
Qwen3.8-27B's exact geometry (24 query heads, 4 KV heads, head dim 256). ASR's
16/8/128 and 16/-/64 do not match any of those routes, so neither is reachable
without new shape support.

Sizing it before building it:

- **Decode (`50.13 ms`, the large one) gets nothing from FlashAttention.** At
  batch one the score "matrix" is a single row of at most 260 values; there is no
  quadratic intermediate to tile away and no matrix-multiply shape to feed the
  WMMA units. It is bound by KV-cache bandwidth and dispatch width, and is
  already `1.77x` improved.
- **Both prefill paths already avoid the quadratic write.** Each keeps its score
  row in LDS and never materializes S to DRAM, which is FlashAttention's actual
  IO win. What is left is using matrix cores for QK and PV, worth at most the
  `15.60 ms` those two kernels cost in total -- `1.6%` of the request, and
  realistically half that.

So a FlashAttention port for this model is capped at roughly `1%` end to end and
would need new CK or AOTriton shape support to reach it. Not worth it while
`747 ms` of the request is decode-projection bandwidth.

## September 2026 route audit

The production audio encoder now returns its device-resident output directly to
the text decoder. Validation keeps only a scalar finite flag on the host; large
diagnostic traces and the device-to-host-to-device transfer remain confined to
focused tests. Audio and text runtime construction also overlap their
independent weight loading.

On the same warm filesystem state, the previous and retained release binaries
produced the exact same transcript:

| Route | Weight load | Resident request mean |
|---|---:|---:|
| Previous release | `548.094 ms` | `992.48 ms` |
| Retained device route | `389.198 ms` | `992.14 ms` |

Two proposed defaults were rejected:

- Mapping decoder weights reduced load time to `332.669 ms`, but increased the
  resident request mean to `1570.89 ms` (`58%` slower). Device copies therefore
  remain the default.
- Restricting audio self-attention to the official 104-token window preserved
  the 15-second transcript but reduced final-encoder cosine to `0.964719`,
  below the `0.995` gate. On a 60.2-second fixture it became non-finite, while
  full attention completed with 180 generated tokens in `4842.8 ms`.

The 104-token experiment was removed completely. There is no audio-window
environment variable, segmented-window route, or dormant attention-window
setting in production.

## Remaining headroom

| Item | Now | Assessment |
|---|---:|---|
| Decode projections | `747.58 ms` | At `91-99%` of the DRAM read ceiling. Only fewer bytes would help, which means quantization and an accuracy loss. |
| Prefill and audio GEMM | `98.58 ms` | Algorithm selection, not a missing kernel -- `1.47x` available, see below. A hand-written WMMA kernel was measured and rejected. |
| Decode attention | `50.13 ms` | 16 blocks on 40 CUs is structural at batch one. Splitting the context would use the whole machine but changes the accumulation order. FlashAttention does not apply at batch one. |
| Residual-add + RMSNorm | `24.64 ms` | 2688 single-block dispatches, latency bound. More threads changes the reduction tree, so not bit-exact. |
| Beating the memory roofline | -- | Speculative decoding with exact verification is the only route that reduces per-token weight traffic without changing greedy output. No draft model exists for this checkpoint. |
