# MiniMax H3 BF16 Residency Baseline

Status: initial gfx1151 loader decision, 2026-08-22

## Contract

The production loader accepts only the committed source manifest with SHA-256
`8776014efafac996761041c0e3df740b41667275ed12a1e021cf8b49faf9b009`.
Inspection parses the two safetensors indexes and every shard header with
bounded `pread`; it maps no tensor payload and performs no HIP allocation.

The immutable BF16/F32 inventory is divided into non-overlapping execution
phases:

| Phase | Tensor bytes | GiB |
| --- | ---: | ---: |
| Qwen prompt encoder | 66,714,780,128 | 62.13 |
| AdaLN/time precompute | 26,142,079,488 | 24.35 |
| DiT core and heads | 40,138,350,656 | 37.38 |
| VisualVAE | 10,415,484,128 | 9.70 |
| AudioVAE | 605,306,340 | 0.56 |

A `PhaseSession` owns its stream, page-aligned file mappings, read-only HIP
registrations or device-copy allocation, persistent arena, and scratch arena.
Destruction and every failed load unwind those resources in reverse order.
Only one phase session may be promoted by the orchestrator at a time, so the
five phase peaks do not sum.

## Initial gfx1151 Measurement

Machine: the supported 128 GB Strix Halo development host, ROCm 7.2.3,
integrated `gfx1151`. Artifact: pinned MiniMax H3 FL2VA BF16 checkpoint.
The focused AudioVAE residency case covers 605,306,340 tensor bytes plus two
16 MiB arenas.

```sh
STRIX_H3_MODEL_ROOT=/var/llms/huggingface/MiniMax-H3 \
STRIX_H3_SOURCE_MANIFEST=src/models/minimax_h3/\
MINIMAX_H3_FL2VA_BF16.source-manifest.json \
./build-h3-166-hip/minimax_h3_runtime_hip_test
```

Observed process runs:

| Cache state/order | Mapped read-only | Device copy |
| --- | ---: | ---: |
| First cold-ish run | 226.9 ms | 98.2 ms |
| Immediate warm run | 73.3 ms | 97.0 ms |

Metadata inspection took 10.6–14.0 ms and read about 376 KB of safetensors
headers. It read zero tensor payload bytes and mapped or allocated zero model
bytes.

The initial default is explicit device copy. Its load time was stable across
the two runs, its compute-access behavior is conventional, and no H3 kernel
profile exists yet to justify direct system-memory reads. Read-only
`hipHostRegisterMapped | hipHostRegisterReadOnly` is retained as an explicit
diagnostic mode: it succeeded on the target and avoids the 605 MB device copy,
but its cold load was sensitive to page residency.

Issue #175 revisits this default through short production-shape numerical
oracles and focused phase/kernel profiles. A complete 50-step video is not a
development benchmark. Metadata or load-only measurements remain scoped to
their phase and are not complete-route performance claims.

## Prompt-Encoder Streaming Measurement

The Qwen phase does not make its 62.13 GiB inventory resident at once. The
text-only encoder loads the 1.45 GiB embedding for lookup and releases it, then
keeps one layer resident while a second layer is copied on a background
transfer stream. Every read-only file registration is removed after its copy.
Only layers 0 through 49 are addressed; vision and deepstack paths fail before
allocation.

For the six-token fox prompt on the same gfx1151 host:

| Run | Layers | Wall time | Peak device bytes | Dispatches |
| --- | ---: | ---: | ---: | ---: |
| first boundary bring-up | 1 | 1.37 s | 1.45 GiB | 17 |
| first complete run | 50 | 21.80 s | 1.88 GiB | 850 |
| warmer complete run with oracle | 50 | 15.06 s | 1.88 GiB | 850 |
| fully warm repeated validation | 50 | 5.83 s | 1.88 GiB | 850 |

Layer 1 and layer 50 were byte-identical to independent Transformers BF16
oracles. The repeated complete execution retained stable output bytes and
dispatch count, with zero registered host bytes after return.

## Full BF16 Denoiser Measurement

The full-denoiser route does not overlap the 62.13 GiB prompt encoder with the
transformer peak. It accepts only the retained layer-50 tensor and immediately
reduces the live prompt boundary to the 63 KiB refined conditioning tensor.

AdaLN/time weights are never admitted as a 24.35 GiB phase. The runtime keeps
the small timestep embedding, then loads one block's 96,768x2,688 projection
and bias, materializes all required schedule rows, transfers those constants
into the matching core session, and releases the projection before proceeding.
All 50 BF16 core blocks remain resident for repeated denoiser evaluations.

For the six-token fox prompt, 256x256x22 geometry, and four evaluations:

| Measurement | Observed |
| --- | ---: |
| text refiner | 0.73–0.78 s |
| AdaLN precompute | 7.30–9.02 s |
| core load and deterministic setup | 8.65–12.66 s |
| first 528-row full forward | 17.3–17.7 s |
| four-step denoise | 67.4 s |
| accounted peak live bytes | 42.41 GiB |
| process swap | 0 bytes |

Each one-block session releases its 16 MiB pinned loader staging and its GEMM
repeat-validation buffer before publication. Retaining either across 50
blocks would waste substantial memory without benefiting timed execution.

The exact route currently uses preallocated host BF16 buffers to cross the 50
independent block streams. No host tensor arithmetic is performed. This
conservative boundary makes cancellation, stable scratch addresses, and
failure cleanup explicit; issue #175 may replace it only after an end-to-end
profile and unchanged quality evidence.

The text-to-video serving presets retain the same phase ordering. `exact`
keeps 50 blocks and 50 fresh evaluations. `fast` keeps 45 blocks and evaluates
the 20-step schedule at reuse interval 2. `aggressive` keeps 40 blocks and
uses interval 3. Reuse adds only one previous BF16 video velocity buffer and
one previous BF16 audio velocity buffer; it does not retain another core or
decoder phase. Complete exact observations belong here only after an
explicitly requested release validation. They are not required for normal
development iterations.

## Rapid Diagnostic Route

The development preset is not a production latency result, but it establishes
the complete prompt → denoiser → selected VisualVAE lifecycle on the supported
machine. A one-time determinism investigation captured two 256x256x22 runs
using all 50 blocks, four fresh evaluations, and selected frames 0, 11, and
21:

| Measurement | Run 1 | Run 2 |
| --- | ---: | ---: |
| total | 159.375 s | 163.654 s |
| prompt encoder | 22.063 s | 21.161 s |
| denoiser | 67.662 s | 67.451 s |
| selected VisualVAE | 47.783 s | 50.017 s |
| peak denoiser bytes | 42.41 GiB | 42.41 GiB |
| peak VisualVAE bytes | 9.38 GiB | 9.38 GiB |
| process swap | 0 bytes | 0 bytes |

All three delivered PPM files and both parameter documents were byte-identical
between those historical runs. A real asynchronous API request for frame 11
returned the same bytes as the direct CLI and completed in 156.087 seconds.
This evidence is retained; it is not rerun as a routine repetition gate.

## Development Measurement Protocol

Issue #175 uses the 512x512, 22-frame, two-step independent denoiser oracle as
its production-shape correctness gate. Focused phase and kernel profiles then
measure the bottleneck that changed. The prompt conditioning, seed,
model/source revisions, geometry, and arithmetic contract remain fixed; the
candidate implementation is the intentional difference.

Development evidence retains:

- inspection bytes and time;
- prompt weight bytes, load, prefetch wait, submit, GPU, dispatch, and peak
  memory values;
- text refiner, AdaLN precompute, core load, each fresh forward, sampler, and
  denoiser memory values;
- process storage I/O, RSS/HWM/swap, page faults, GPU clocks, and power when
  exposed by the kernel;
- independent teacher/native velocity and final-latent errors at 1,872 rows.

Each required diagnostic is invoked once. A passing full route is not rerun,
and a 50-step generation is never launched merely to obtain performance
statistics.

`tools/strix/h3_cache_control.py` provides the separate reproducible
file-cache-cold preparation. It issues `POSIX_FADV_DONTNEED` for every regular,
non-symlink file under the operator-supplied model root, using `O_NOFOLLOW` and
the opened descriptor's size. It records attempted file/byte counts, errors,
elapsed time, and before/after `/proc/meminfo` snapshots without retaining
paths. Because this is an advisory Linux operation, reports call the result
`eviction-requested`, not a kernel-guaranteed cache purge. It is an optional
diagnostic utility and is not invoked by the issue #175 development
benchmark.

## First Optimization Retention Budget

The row-parallel full-attention candidate is admitted only if the short
production-shape comparison proves all of the following against the explicit
scalar fallback:

- harness hash, kernel release, prompt hash, seed, immutable model/reference
  revisions, and the frozen `exact` preset parameters match;
- binary, harness, prompt, and oracle identities are complete lowercase
  SHA-256 values rather than unvalidated labels;
- analytic full-attention parity covers the production sequence shapes used
  by dev (528 rows), aggressive (780), fast (1,088), and exact (1,872), and
  the frozen preset contract tests pass;
- the candidate improves its measured bottleneck and the fixed two-step
  production-shape workload by at least 5%;
- both successful observations contain the complete required timing, memory,
  I/O, power, clock, swap, and energy telemetry;
- candidate accounted peak bytes and process high-water RSS do not regress by
  more than 2%, and process/system swap remain zero in both observations;
- the candidate 10th-percentile GPU clock sampled under substantial GPU power
  remains within 10% of the scalar observation, and both runs have enough loaded
  samples to distinguish sustained work from launch and teardown;
- GPU power, clocks, and integrated energy are retained as diagnostics, not
  efficiency gates. A faster kernel may legitimately use more of the
  platform's available graphics power.

Fast and aggressive are validated through frozen configuration and
production-shape arithmetic gates. Delivery-level quality and performance can
be measured later when that specific evidence is requested.

This project optimizes interactive latency rather than an efficiency preset.
Temperature is not a comparison objective or pass/fail input because ambient,
cooling, and room conditions vary between operators. The implementation may
use the machine's configured 120–140 W graphics envelope when that power
corresponds to measured useful work. Sustained-clock regression still detects
actual throttling. Busy-waiting, redundant transfers, or recomputation do not
qualify: phase and kernel profiles must explain the latency improvement, and
energy per completed generation remains visible in the report. The original
scalar kernel remains selectable through
`--attention-kernel scalar`; its policy is recorded in parameters and
telemetry.

## Regression Gates

- Duplicate JSON keys, unsafe shard paths, unknown indexed tensors, bad dtypes,
  shape/byte mismatches, overlaps, missing shards, and wrong source-manifest
  bytes fail before payload mapping.
- Failure injection covers target validation, stream creation, mapping,
  prefault, registration/copy, synchronization, and arena allocation.
- Cancellation before publication releases all resources.
- Entering timed execution performs no allocation or first-touch operation.
- The real-checkpoint HIP test remains opt-in through explicit model and
  manifest environment variables; normal CI never downloads weights.
- Exact tokenizer cases include Unicode, contractions, emoji, added tokens,
  empty prompts, malformed UTF-8, unsupported added-token policies, and wrong
  normalizers.
- Analytic gfx1151 tests cover RMSNorm, per-head Q/K normalization, RoPE,
  causal GQA, residual add, and SwiGLU.
- External prompt tests cover selected and final layer boundaries, non-finite
  rejection, stable repeated bytes/dispatches, and registered-page cleanup.
