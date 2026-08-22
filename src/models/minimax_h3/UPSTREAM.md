# MiniMax H3 Upstream Provenance

## Model checkpoint

```text
repository: MiniMaxAI/MiniMax-H3
revision:   42ed227ee7df40d41602854ae760620d6eb651fe
partition:  FL2VA
```

The checkpoint is an external operator-supplied artifact and is not
distributed by Strix-Halo.cpp. Its public license metadata and acquisition
boundary are recorded in `docs/MINIMAX_H3.md` and
`THIRD_PARTY_NOTICES.md`.

## Native implementation reference

```text
repository: antirez/h3.c
revision:   8974cc055ea9c02fcd14cc27dfda3e1027c05153
license:    MIT
copyright:  Copyright (c) 2026 Salvatore Sanfilippo
```

The pinned source supplies the initial independent implementation contract for:

- safetensors tensor names and component boundaries;
- tokenizer and Qwen3-VL layer-50 prompt encoding;
- packed text/video/audio sequence layout;
- sigma schedule, RNG, Euler integration, and core reuse;
- the 50-block Omni Transformer;
- VisualVAE and AudioVAE decoding;
- selected-frame development and audiovisual mux behavior;
- exact, fast, and aggressive performance references.

Metal, MPSGraph, Objective-C runtime integration, command-line presentation,
and Apple-specific resource management are not imported into the Linux
runtime.

## TensorOps ancestry

The pinned h3.c `THIRD_PARTY_NOTICES.md` states that the rectangular Morton
decoder and dynamic symmetric INT8/Metal TensorOps scheduling design in
`h3_shaders.metal` are adapted from ccv's `NAMatMulKernel` and
`NAInt8MatMulKernel` under BSD-3-Clause:

```text
Copyright (c) 2010, Liu Liu
All rights reserved.
```

If source expression or a recognizably adapted implementation from those
sections is brought into the gfx1151 backend, the BSD-3-Clause notice and
conditions must remain with the affected source and binary distribution.

## Adaptation record

Every copied or adapted file must record:

- upstream path and pinned revision;
- local destination;
- whether it was copied, translated, or independently reimplemented;
- material local changes;
- applicable MIT and, where relevant, BSD-3-Clause notices.

The model-private H3 implementation lives under `src/models/minimax_h3/`.
Shared engine infrastructure must not gain H3 tensor-name guesses or
model-specific numerical dispatch.

Current prompt-boundary adaptations:

| Upstream source | Local destination | Method and material changes |
| --- | --- | --- |
| `h3_tokenizer.m` | `tokenizer.cpp`, `tokenizer.hpp` | Semantic translation to C++20; Foundation storage replaced by strict project JSON and ICU NFC/category APIs; stronger duplicate/range/schema rejection |
| `h3_text_encoder.c` | `prompt_encoder.hip`, `prompt_encoder.hpp` | Layer/tensor contract adapted; Metal resource management replaced by device-copy streaming, next-layer prefetch, hipBLASLt, phase lease, cancellation, and telemetry |
| text kernels in `h3_shaders.metal` | `prompt_encoder_ops.cuh` | Operation-boundary translation to HIP; BF16 rounding and FP32 reductions retained; no Metal/MPSGraph or ccv TensorOps code imported |
| `tests/test_tokenizer.c`, `tests/test_real_prompt.c` | MiniMax H3 tokenizer and prompt-encoder tests | Released corpus retained; synthetic malformed cases, analytic HIP fixtures, content-hashed external Transformers boundaries, repetition, and leak checks added |

Current sampler-boundary adaptations:

| Upstream source | Local destination | Method and material changes |
| --- | --- | --- |
| `h3_host.c`, `h3_host.h` | `sampling.cpp`, `sampling.hpp` | Semantic C++20 translation of checked geometry, temporal alignment, text-only MM-RoPE layout, shifted serving schedules, timestep rows, PCG/Box-Muller noise, and explicit ownership/error handling |
| packing and reuse helpers in `h3_dit.c` | `sampling.cpp`, `sampling.hpp` | Exact visual/audio row order, reuse selection, and bounded extrapolation retained; Ref2VA and frame-anchor segment kinds deferred |
| `h3_euler_bf16` in `h3_shaders.metal` | `sampling.hip` | Operation-boundary translation to one gfx1151 HIP kernel; F32 sample state, BF16 velocities, fused extrapolation/update, bounds validation, and no CPU tensor fallback |
| `tests/test_h3.c`, `tests/test_bf16.c` | MiniMax H3 sampling host/HIP tests | Exact schedule/layout/row-map/noise hashes, temporal cases, overflow/malformed inputs, round trips, reuse masks, and repeated byte-exact device Euler coverage |

Current DiT-boundary adaptations:

| Upstream source | Local destination | Method and material changes |
| --- | --- | --- |
| `h3_dit.c` block load/run paths | `dit.hpp`, `dit.cpp`, `dit.hip` | One-block BF16 session with model-private tensor contracts, direct shard loading, atomics-disabled rocBLAS projections warmed before execution, fixed activation storage, explicit stream/cancellation/telemetry, and no allocation during block execution |
| DiT kernels in `h3_shaders.metal` | `dit_ops.cuh` | Operation-boundary HIP translation for casts, residual arithmetic, SiLU/SwiGLU, RMS/layer/AdaLN, gated residuals, grouped QKV interpretation, per-head Q/K norm, partial 3D MM-RoPE, and full SDPA |
| `tests/test_real_dit_block.c` | `minimax_h3_dit_hip_test.hip`, `tools/strix/h3_dit_golden.py` | Odd-tail analytic fixtures plus an operator-owned 528-row block-0 oracle produced by direct ROCm PyTorch formulas; retained payloads are never committed |

Current full-denoiser adaptations:

| Upstream source | Local destination | Method and material changes |
| --- | --- | --- |
| `h3_dit_schedule.c` | `denoiser.hpp`, `denoiser.hip` | Independent C++/HIP implementation of unique video/audio timestep rows, F32 timestep embedding, streamed per-block AdaLN projection, final modulation, and explicit phase telemetry |
| token refinement, patch packing, core loop, and final heads in `h3_dit.c` | `denoiser.hip` | Complete text-only 50-block BF16 route; all core weights resident, fixed activation arenas, full attention, modality maps, F32 final heads, BF16 velocity boundary, cancellation after every block/step, and no CPU tensor compute |
| GPU Euler denoise path in `h3_dit.c` | `denoiser.hip`, `sampling.hip` | Explicit development/serving/exact evaluation counts, active block prefixes, whole-velocity reuse with two BF16 boundaries, independent video/audio deltas, bounded extrapolation, and device-resident F32 samples |
| `tests/test_real_dit.c`, `tests/test_semantic_dit.c` | `minimax_h3_denoiser_hip_test.hip`, `tools/strix/h3_denoiser_golden.py` | Tiny lifecycle smoke, 528-row full velocity/four-step latent oracle, retained 512x512x22x50 route, byte-repeat, cancellation/reuse, memory, page-fault, and zero-swap evidence |

Current output-decoder adaptations:

| Upstream source | Local destination | Method and material changes |
| --- | --- | --- |
| VisualVAE load, temporal chunk, tiling, and selected-frame paths in `h3_video_vae.c` | `video_vae.hpp`, `video_vae.cpp`, `video_vae.hip` | C++20/HIP phase with exact released F32 weights, deterministic 256–320 pixel tile planning, seven-latent/22-frame chunks, bounded spatial/temporal output composition, selected-frame pruning, cancellation, and reusable scratch |
| VisualVAE kernels in `h3_shaders.metal` | `video_vae_ops.cuh` | Independent HIP implementation of latent normalization, patch embedding, RMS/layer norm, QKV normalization/RoPE, full attention, SwiGLU, residual scaling, output projection, and selected RGB unpack |
| `h3_audio_vae.c` decoder path | `audio_vae.hpp`, `audio_vae.cpp`, `audio_vae.hip` | Released F32 decoder only; exact tensor contracts, resident normalized convolution weights, stereo-as-batch execution, seven BigVGAN stages, cancellation, byte-repeat, and memory/fault/swap telemetry; unused encoder attention is excluded |
| AudioVAE kernels in `h3_shaders.metal` | `audio_vae_ops.cuh` | Independent HIP implementation of latent normalization, weight normalization, Conv1d, ConvTranspose1d, fused alias-free SnakeBeta, residual accumulation, clipping, and channel-major PCM output |
| `h3_ffmpeg.c`, `tests/test_av_mux.c` | `media.hpp`, `media.cpp`, `minimax_h3_media_test.cpp` | C++20 two-pipe FFmpeg process with bounded PCM interleave staging, nonblocking cancellation-aware writes, deterministic cleanup, headless Nix dependency, FFprobe codec/geometry/rate/duration/sync validation, and no temporary uncompressed media file |
| `tests/test_real_video_vae.c`, `tests/test_real_audio_vae.c` | VisualVAE/AudioVAE HIP tests and direct PyTorch golden tools | Operator-owned frozen latent/frame/waveform/spectrogram payloads, analytic primitive fixtures, external hashes, selected/full parity, stereo separation, clipping, cancellation, repeatability, and zero-swap gates |

Current text-only orchestration adaptations:

| Upstream source | Local destination | Method and material changes |
| --- | --- | --- |
| generation flow and exact/fast/aggressive examples in `h3.c`, `README.md` | `generation.hpp`, `generation.cpp`, `video_cli.hpp`, `video_cli.cpp` | Model-private C++ phase orchestrator and direct CLI with pinned exact/fast/aggressive/dev parameter contracts, prompt hashing, explicit overrides, selected-frame rapid iteration, progress/cancellation, atomic reports/outputs, and no implicit cache or model acquisition |
