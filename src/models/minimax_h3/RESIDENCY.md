# MiniMax H3 BF16 Residency Baseline

Status: retained second-profile implementation, 2026-08-23

## Contract

The production loader accepts only the committed source manifest with SHA-256
`00a83367b87017f1f3d5547a963b20567a46d7aea44f0aac495c816147720d89`.
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
GUFO_H3_MODEL_ROOT=/var/llms/huggingface/MiniMax-H3 \
GUFO_H3_SOURCE_MANIFEST=src/models/minimax_h3/\
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

Exact-mode construction validates the four fixed BF16 GEMM shapes once, then
loads two independent DiT blocks concurrently while AdaLN projections are
precomputed. Concurrency is deliberately bounded at two workers. The 528-row
oracle reduced measured setup from approximately 35.0 s to 13.51 s while
retaining four plan validations, zero swap, and the refined-text, modulation,
video-velocity, and audio-velocity quality bounds.

Direct file-backed execution was rejected for repeatedly reused DiT weights:
although its output passed the block oracle, a production 1,872-row block
regressed against normal device-resident execution. A phase-wide
mapped-to-device copy also increased measured setup. Mapped weights therefore
remain appropriate for streamed phases such as the prompt encoder, not for the
repeatedly reused DiT core.

Production-shape profiling pins four rocBLAS solutions specifically for the
1,872-row 512x512 workload. The focused block improved from 203.28 ms to
177.33 ms (12.8%). The independent 512 oracle retained its refined-text,
block-0 modulation, video-velocity, and audio-velocity bounds.

VisualVAE linear bias application uses aligned `float4` transactions whenever
the output width permits it. The selected-frame oracle remained within
`8.16e-7` relative L2 and `2.39e-6` relative maximum error; profiled bias
kernels fell from 1.549 s to 0.677 s (56.3%).

AdaLN/time weights are never admitted as a 24.35 GiB phase. The runtime keeps
the small timestep embedding, then loads one block's 96,768x2,688 projection
and bias, materializes all required schedule rows, transfers those constants
into the matching core session, and releases the projection before proceeding.
All 50 BF16 core blocks remain resident for successive denoiser evaluations.

The pre-issue-175 cross-block validation executed one forward only:

| Geometry | Rows | AdaLN | Core load | Forward | Peak | Swap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256x256x22 | 528 | 11.127 s | 21.981 s | 12.275 s | 42.50 GiB | 0 |
| 512x512x22 | 1,872 | 5.839 s | 25.736 s | 57.265 s | 59.90 GiB | 0 |

Each one-block session releases its 16 MiB pinned loader staging and its GEMM
validation buffer before publication. Retaining either across 50 blocks would
waste substantial memory without benefiting timed execution.

The exact route now keeps the 50-block hidden state on one ordered HIP stream.
Two reusable BF16 device buffers ping-pong between blocks, and one device row
map is uploaded per evaluation. The block sessions still own independent,
stable scratch arenas, but no longer copy the approximately 20 MiB hidden
state to and from host memory or synchronize after every block. Cancellation
is checked before each block submission, and the final velocity heads provide
the single forward synchronization boundary.

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

Decoder-local changes instead use the frozen 256x256 selected-frame VisualVAE
teacher and one profiled tile. They do not require another denoiser execution
or MP4 generation.

Each required diagnostic is invoked once. A passing full route is not rerun,
and a full exact generation is never launched merely to obtain performance
statistics.

## Issue #184 Second Focused Profile

The retained second pass used one fresh uninstrumented workload and one final
profile for each affected phase. No complete generation, denoising trajectory,
or statistical repetition was run.

| Phase | Fresh baseline | Retained | Reduction | Speedup | Peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| AudioVAE, 37 latent frames to stereo 29,600-sample PCM | 59.4383 s | 2.42326 s | 95.92% | 24.53x | 0.286 GiB |
| VisualVAE, one 256x256x22 five-selected-frame tile | 8.60171 s | 6.12212 s | 28.83% | 1.405x | 9.766 GiB |
| DiT, one 512x512x22 1,872-row 50-block forward | 9.02751 s | 5.49537 s | 39.13% | 1.643x | 58.880 GiB |

All three retained invocations used zero swap. AudioVAE's peak grew from
0.250887 GiB to 0.286173 GiB because one bounded, reusable im2col workspace
replaces scalar convolution. VisualVAE's peak was unchanged. The denoiser's
0.077 GiB increase is the two device-resident hidden buffers plus per-block
normalization factors; it is 0.13% of the phase peak and replaces equivalent
host activation storage.

### AudioVAE convolution inventory

The 37-frame decoder executes 129 Conv1d calls and seven ConvTranspose1d
calls. Every convolution remains F32 input, weight, accumulation, and output.

| Boundary | Output length | Channels | Kernel / stride | Conv1d calls |
| --- | ---: | ---: | --- | ---: |
| latent input projection | 37 | 32 -> 2,048 | 1 / 1 | 1 |
| decoder pre-convolution | 37 | 2,048 -> 1,024 | 7 / 1 | 1 |
| stage 0 | 185 | 1,024 -> 512 | transpose 9 / 5 | 18 |
| stage 1 | 925 | 512 -> 256 | transpose 9 / 5 | 18 |
| stage 2 | 1,850 | 256 -> 128 | transpose 4 / 2 | 18 |
| stage 3 | 3,700 | 128 -> 64 | transpose 4 / 2 | 18 |
| stage 4 | 7,400 | 64 -> 32 | transpose 4 / 2 | 18 |
| stage 5 | 14,800 | 32 -> 16 | transpose 4 / 2 | 18 |
| stage 6 | 29,600 | 16 -> 8 | transpose 4 / 2 | 18 |
| decoder post-convolution | 29,600 | 8 -> 1 | 7 / 1 | 1 |

Each stage's 18 residual convolutions are three blocks with kernels 3, 7, and
11. Each block has three first convolutions at dilations 1, 3, and 5 and three
unit-dilation second convolutions. The retained implementation lowers both
normal and transposed convolutions to F32 im2col plus rocBLAS GEMM, reorders
transposed weight normalization once at load, and reuses the maximum required
column buffer. It also splits alias-free SnakeBeta into its released
upsample/activation and downsample operations without changing the filter or
sine sequence.

The final AudioVAE profile is no longer convolution-dominated: SnakeBeta
upsampling accounts for 43.93% of kernel time, im2col 24.36%, and SnakeBeta
downsampling 18.23%. The waveform gate measured maximum error
`6.03162e-5` and relative L2 `1.08936e-5`; the spectrogram gate measured
relative L2 `3.62714e-6` and relative maximum `5.8276e-6`.

### VisualVAE attention preparation

VisualVAE now computes Q and K norm reductions once per head and broadcasts
the factors. QKV and projection biases are folded into kernels only after the
same F32 GEMM rounding boundary, output/MLP biases are fused into their
existing residual/SwiGLU operations, and the stable F32 softmax reduction
order remains unchanged. The final profile attributes 35.99% to stable
softmax, 27.59% to GEMMs, and 8.69% to QKV preparation; vectorized SwiGLU fell
to 0.241 s.

The selected-frame gate measured relative L2 `8.14968e-7`, relative maximum
`2.24925e-6`, and maximum absolute error `1.6503e-6`.

### Device-resident DiT and split normalization

The first retained denoiser candidate removed the per-block host round trips,
but improved the full forward by only about 3%, proving that the custom
normalization kernels were the larger problem. The profile showed the
single-kernel grouped QKV path spilling 1,240 bytes per thread and AdaLN
spilling 436 bytes per thread.

The final path preserves the same F32 reduction order but separates factor
calculation from BF16 application. Grouped Q/K inverse factors are produced by
one wave per head and consumed by a no-spill RoPE/application kernel. AdaLN
similarly uses a reduction kernel followed by a packed application kernel.
The factors share one reusable F32 buffer per block. The DiT translation unit
uses `-O2`; `-O3` was rejected because it retained the spill-heavy allocation,
while the preset's unqualified compile mode was not optimized.

In the final profile, grouped QKV factor and application kernels total
0.140 s instead of 2.236 s, and core AdaLN reduction/application kernels total
0.031 s instead of 1.584 s. The next profiled hotspots are SwiGLU, fused
attention, and the dense GEMMs; they are not required to meet this issue's
quality-preserving target.

The production oracle remained green: refined text relative L2/max
`0.00469088`/`0.00167411`, block-0 modulation
`0.0022895`/`0.00478469`, video velocity
`0.0132707`/`0.0154553`, and audio velocity
`0.00731056`/`0.0149873`.

The original one-block profile identified dense attention as the actual
bottleneck. The retained progression is:

| 1,872-row block measurement | Original | First batched path | Final fused path |
| --- | ---: | ---: | ---: |
| block GPU time | 38.518 s | 1.076 s | 88.279 ms |
| relative speedup | 1.00x | 35.79x | 436.32x |

The first profile attributed 31.640 of 32.226 profiled GPU seconds to the
custom dense-attention kernel; all GEMMs together took 0.207 seconds.
Replacing QK and PV with strided-batched rocBLAS GEMMs and keeping softmax
explicit first removed that bottleneck. Parallelizing the per-row max,
exponential, and sum reduced the intermediate block from 3.477 seconds to
1.076 seconds.

The issue #175 gfx1151 path fused QK, scaled softmax, and PV through Composable
Kernel's BF16 WMMA attention kernel. It no longer materializes the 1.097 GiB
F32-score/BF16-probability workspace for the supported production shape;
unsupported shapes retain a checked fallback that owns those matrices only
when selected. SwiGLU and residual gates process eight BF16 values per thread.
Two alternative QKV layouts were rejected because they preserved the oracle
but increased production-shape latency. Issue #184 subsequently replaced the
spill-heavy grouped QKV and AdaLN kernels and changed the translation unit to
`-O2`, as recorded above.

The 37,716-row full-resolution shape now has a separately retained gfx1151
Triton attention kernel. The packaged AOTriton dispatch took `1,313.937 ms`
and allocated 416 private bytes per thread in the captured trace. The
shape-specialized 64x32 kernel took `1,268.501 ms`, a `3.46%` attention-kernel
reduction, with 24,576 bytes of LDS and zero private scratch. It writes the
row-major BF16 output consumed by the following projection directly. Other
row counts and any module-load failure continue through the existing
AOTriton/Composable Kernel paths, so lower-resolution behavior is unchanged.

The build compiles and embeds the HSACO reproducibly from the tracked Triton
source; generated binaries are not stored in Git. The compiler step rejects
the kernel if AMDGPU metadata reports any VGPR/SGPR spill or private segment.
Against the original full block, 5,379 of 202,761,216 BF16 elements differed
(`0.002653%`), with relative L2 `2.19037e-5` and relative max `0.00125628`.
The kernel result implies about `2.27 s` saved per 50-block forward, or about
`111 s` across 49 evaluations; that extrapolation was not validated by
running another full video generation.

The final 528-row teacher comparison measured block-output relative L2
`0.00461029` and relative max `0.00483092`, both below the frozen `1e-2`
limits. Its attention AdaLN, attention output, and MLP AdaLN boundaries also
remain below their limits. No 50-block forward or denoising evaluation was
needed for this block-local change.

The historical issue #175 timed block contains eleven dispatches whose trace
duration sums to 88.225 ms, matching the 88.279 ms runtime event. The largest
remaining dispatches are the second MLP GEMM at 22.594 ms, first MLP GEMM at
21.881 ms, QKV projection at 17.022 ms, fused attention at 16.428 ms, and
output projection at 7.527 ms. All custom normalization, RoPE, SwiGLU, and
gate dispatches together take about 3.0 ms. This shifted work from scalar
attention arithmetic to dense-projection selection and block-level execution.

The accompanying short power capture peaked at 44 W and 1,192 MHz. That is a
diagnostic of the 92.494 ms wall-time observation, not a utilization ceiling:
the workload completes before the APU power controller can ramp toward the
configured 120--140 W envelope. Extending or repeating the workload merely to
raise the wattage would add work without improving user latency.

The explicitly requested aggressive smoke re-ranked the optimized route:
VisualVAE used `188.650` of `229.382` seconds (`82.24%`), while the denoiser
used `12.158` seconds (`5.30%`). Its focused decoder profile attributed
`42.815` of `46.685` GPU seconds to scalar full attention.

The retained VisualVAE path materializes one 394.2 MiB F32 score/probability
matrix and executes QK and PV through strided-batched rocBLAS GEMMs with an
in-place stable F32 softmax. Exact-shape solution indices also tune the dense
QKV, output, and MLP projections. The frozen tile fell from `46.6852` seconds
to `4.79063` seconds after attention replacement and finally to `2.82558`
seconds, a `16.522x` scalar-to-final speedup. Selected-frame relative L2
remained `8.15249e-7`. Phase peak increased from 9.38 GiB to 9.77 GiB, swap
remained zero, and sampled package power reached `134 W`. The profile is now
dominated by useful matrix work rather than a scalar attention loop.

Alternative attention solution indices and a register-cached softmax were
measured once and removed because their different F32 reduction order failed
the frozen output gate. Performance alone does not admit them.

`tools/gufo/h3_cache_control.py` provides the separate reproducible
file-cache-cold preparation. It issues `POSIX_FADV_DONTNEED` for every regular,
non-symlink file under the operator-supplied model root, using `O_NOFOLLOW` and
the opened descriptor's size. It records attempted file/byte counts, errors,
elapsed time, and before/after `/proc/meminfo` snapshots without retaining
paths. Because this is an advisory Linux operation, reports call the result
`eviction-requested`, not a kernel-guaranteed cache purge. It is an optional
diagnostic utility and is not invoked by the issue #175 development
benchmark.

## Optimization Retention and Completion Budget

The batched full-attention candidate is admitted only if the focused
single-block evidence proves all of the following against the retained
baseline:

- harness hash, kernel release, prompt hash, seed, immutable model/reference
  revisions, and the frozen `exact` preset parameters match;
- binary, harness, prompt, and oracle identities are complete lowercase
  SHA-256 values rather than unvalidated labels;
- the 528-row fallback parity test, independent block teacher, frozen preset
  contract tests, and one 1,872-row retained-path profile pass;
- an intermediate candidate improves its measured bottleneck and the fixed
  1,872-row block workload by more than the 5% measurement-noise floor;
- the retained observation contains the required timing and memory telemetry;
- unexplained candidate peak-byte or high-water-RSS regressions above 2% are
  rejected, and process/system swap remain zero. An explicit workspace may be
  retained when its size is accounted exactly and the profile proves a
  structural latency reduction; the VisualVAE's 394.2 MiB F32 attention matrix
  qualifies with a `16.522x` fixed-tile improvement;
- GPU power, clocks, and integrated energy are retained as diagnostics, not
  efficiency gates. A short faster kernel may finish before clocks and power
  ramp, while sustained useful work may legitimately use all of the platform's
  configured graphics envelope.

The 5% floor rejects ambiguous measurements; it is not the optimization
target or a reason to close the issue. Issue #175 targets structural,
double-digit reductions in the production-shape block. Work continues while
the profiler identifies a tractable dominant bottleneck; a small intermediate
win is useful only when it preserves quality and advances that larger result.

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
