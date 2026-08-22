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
oracles and focused phase/kernel profiles. A complete exact video is not a
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
| current complete oracle invocation | 50 | 21.38 s | 1.88 GiB | 850 |

Layer 1 and layer 50 were byte-identical to independent Transformers BF16
oracles. The current complete invocation left zero registered host bytes after
return. Routine quality validation does not repeat this 50-layer phase.

## Full BF16 Denoiser Measurement

The full-denoiser route does not overlap the 62.13 GiB prompt encoder with the
transformer peak. It accepts only the retained layer-50 tensor and immediately
reduces the live prompt boundary to the 63 KiB refined conditioning tensor.

AdaLN/time weights are never admitted as a 24.35 GiB phase. The runtime keeps
the small timestep embedding, then loads one block's 96,768x2,688 projection
and bias, materializes all required schedule rows, transfers those constants
into the matching core session, and releases the projection before proceeding.
All 50 BF16 core blocks remain resident for successive denoiser evaluations.

Current cross-block validation executes one forward only:

| Geometry | Rows | AdaLN | Core load | Forward | Peak | Swap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256x256x22 | 528 | 11.127 s | 21.981 s | 12.275 s | 42.50 GiB | 0 |
| 512x512x22 | 1,872 | 5.839 s | 25.736 s | 57.265 s | 59.90 GiB | 0 |

Each one-block session releases its 16 MiB pinned loader staging and its GEMM
validation buffer before publication. Retaining either across 50 blocks would
waste substantial memory without benefiting timed execution.

The exact route currently uses preallocated host BF16 buffers to cross the 50
independent block streams. No host tensor arithmetic is performed. This
conservative boundary makes cancellation, stable scratch addresses, and
failure cleanup explicit; issue #175 may replace it only after an end-to-end
profile and unchanged quality evidence.

The text-to-video serving presets retain the same phase ordering. `exact`
keeps 50 blocks and performs 49 fresh evaluations over Diffusers' 50 sigma
points including terminal zero. `fast` keeps 45 blocks and performs 19
evaluations over 20 points at reuse interval 2. `aggressive` keeps 40 blocks
and uses 19 evaluations with interval 3. The thinned routes score every
candidate block's schedule-dependent AdaLN gates, protect blocks 0, 1, and 49,
and retain the 45 or 40 highest-value blocks in original execution order.
These are Diffusers-schedule serving modes with h3.c-inspired thinning and
reuse; they are not exact reproductions of h3.c's 20-transition presets.
For normal 19-evaluation schedules, at most 512 MiB of temporary host
modulation data is retained to avoid recomputing selected projections; larger
custom schedules use a bounded two-pass path. Reuse adds only one previous F32
video velocity buffer and one previous F32 audio velocity buffer; it does not
retain another core or decoder phase. Complete exact observations belong here
only after an explicitly requested release validation. They are not required
for normal development iterations.

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

Issue #175 uses one independently frozen 528-row block-0 teacher as its
correctness gate and one deterministic 1,872-row block as its performance
workload. Focused phase and kernel profiles measure only the bottleneck that
changed. The model/source revisions, block weights, inputs, and arithmetic
contract remain fixed; the candidate implementation is the intentional
difference.

Development evidence retains:

- inspection bytes and time;
- block load, GPU and wall time, output hash, and workspace bytes;
- process storage I/O, RSS/HWM/swap, page faults, GPU clocks, and power when
  exposed by the kernel;
- independent teacher/native retained-boundary errors at 528 rows.

Each required diagnostic is invoked once. A passing full route is not rerun,
and a full exact generation is never launched merely to obtain performance
statistics.

The original one-block profile identified dense attention as the actual
bottleneck:

| 1,872-row block measurement | Original | Retained batched path |
| --- | ---: | ---: |
| block GPU time | 38.518 s | 1.076 s |
| relative speedup | 1.00x | 35.79x |

The first profile attributed 31.640 of 32.226 profiled GPU seconds to the
custom dense-attention kernel; all GEMMs together took 0.207 seconds.
Replacing QK and PV with strided-batched rocBLAS GEMMs and keeping softmax
explicit removes that bottleneck. The shared F32 score and BF16 probability
workspaces total 1.097 GiB at 1,872 rows and are reused by every block.
Parallelizing the per-row max, exponential, and sum then reduced the retained
block from 3.477 seconds to 1.076 seconds.

The retained 528-row teacher comparison measured final block-output relative
L2 `0.00476406` and relative max `0.00956938`, both below the frozen `1e-2`
limits. No 50-block forward or denoising evaluation was needed for this
attention-local change.

The final profile ranks parallel softmax at 435.7 ms (31.49% of captured
kernel time), grouped QKV normalization/RoPE at 390.3 ms (28.21%), and rocBLAS
GEMMs at 208.9 ms (15.10%). These measured phases define the next optimization
order.

Historical scalar phase timing ranks the denoiser at 98.326% of complete
latency, VisualVAE at 1.240%, prompt encoding at 0.163%, and AudioVAE at
0.016%. After retaining row-parallel attention, the DiT forward remains the
dominant target. Dense projection/MLP work and the conservative host-staged
block boundary are the next candidates, but neither is authorized without a
new focused profile.

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

The batched full-attention candidate is admitted only if the focused
single-block evidence proves all of the following against the retained
baseline:

- harness hash, kernel release, prompt hash, seed, immutable model/reference
  revisions, and the frozen `exact` preset parameters match;
- binary, harness, prompt, and oracle identities are complete lowercase
  SHA-256 values rather than unvalidated labels;
- the 528-row fallback parity test, independent block teacher, frozen preset
  contract tests, and one 1,872-row retained-path profile pass;
- the candidate improves its measured bottleneck and the fixed 1,872-row
  block workload by at least 5%;
- the retained observation contains the required timing and memory telemetry;
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
telemetry. The scalar implementation remains available through the explicit
`row_parallel_attention=false` diagnostic fallback.

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
  rejection, exact teacher bytes/dispatches, and registered-page cleanup.
