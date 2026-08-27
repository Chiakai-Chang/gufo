# MiniMax H3 Upstream Provenance

## Model checkpoint

```text
repository: MiniMaxAI/MiniMax-H3
revision:   42ed227ee7df40d41602854ae760620d6eb651fe
partition:  FL2VA
```

The checkpoint is an external operator-supplied artifact and is not
distributed by gufo. Its public license metadata and acquisition
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

The released Hugging Face Diffusers implementation is the authoritative
quality-parity reference for the model's Python pipeline:

```text
repository: huggingface/diffusers
revision:   fe15005a333d1270b490e4885a6ea13b66b1092a
```

A line-by-line text-only audit covers the pipeline, transformer, scheduler,
noise construction, VisualVAE, and AudioVAE. Where h3.c and Diffusers differ,
the native exact path follows Diffusers when the difference changes model
arithmetic or decoder context.

### Audited implementation differences

| Area | Pinned Diffusers authority | Pinned h3.c behavior | Native Strix Halo decision |
| --- | --- | --- | --- |
| exact schedule | 50 sigma points including terminal zero, 49 forwards | 20 transitions in its accelerated examples | `exact` is 50/49 |
| fast/aggressive schedule | no released 45/40-block preset contract | ranks gates on a 21-point/20-transition schedule | native 20-point/19-forward serving modes borrow ranking/reuse only and never claim exact h3.c preset parity |
| Euler arithmetic | F32 velocity, sigma reconstructed from F32 timestep, materialized denoised estimate and blend | BF16 velocity boundary and direct grid-delta arithmetic | F32 sample/velocity with Diffusers' two-stage operation order |
| initial noise | one PyTorch generator consumed by video then channel-major audio | separate restarted PCG streams | one sequential PCG/Box-Muller stream; injected frozen tensors, not equal integer seeds, are the cross-runtime oracle because PCG values differ from PyTorch |
| timestep embedding | F32 MLP and SiLU before BF16 AdaLN projection | BF16 boundary before the final activation | F32 through SiLU, then BF16 projection input |
| spatial MM-RoPE | aspect-normalized released coordinates | applies a `0.5` spatial heuristic at 256x256 | standard Diffusers coordinates for every exact/native preset |
| block thinning | full 50-block released model | protects blocks 0, 1, and 49 and gate-ranks the remainder | exact uses all 50; fast/aggressive reuse h3.c's ranking algorithm on their own declared schedule |
| attention implementation | PyTorch SDPA | Metal full attention | native HIP row-parallel full attention, with scalar diagnostic fallback, bounded by the independent block/full-forward oracles |
| VisualVAE spatial context | fixed 256-pixel tiles | dynamically selects 288/320-pixel tiles | fixed 256-pixel tiles in the quality route |
| decoder precision | released CUDA VisualVAE uses FP16 autocast | F32 decoder implementation | F32 VisualVAE and AudioVAE quality route; comparisons use numerical bounds rather than byte identity with CUDA |
| RGB quantization | NumPy ties-to-even | `lrintf`, ties-to-even under the default rounding mode | explicit deterministic ties-to-even; covered by a byte-level PPM test |
| output resize | pipeline/output-processor dependent | vImage high-quality scaling | FFmpeg Lanczos for fast/aggressive output-canvas scaling; exact performs no resize |
| request duration | released pipeline accepts aligned 5--15 second requests | short native development examples | native additionally supports a 22-frame diagnostic/serving extension; it is not described as the released request envelope |
| prompt length | pipeline/model tokenizer contract | implementation-specific | text-only serving deliberately caps prompts at 4,096 tokens |
| conditioning modes | FL2VA plus first/last/reference pipeline layouts | corresponding reference modes exist | only text-only FL2VA is admitted; first frame, last frame, ordered references, and Ref2VA remain explicit future work |

The audit found no remaining semantic difference in prompt presentation and
layer-50 capture; raw grouped-QKV head packing; the converter's deliberate
SwiGLU half swap versus raw checkpoint gate/up order; packed text/audio/video
row order and modality tags; video patchification; channel-major audio
packing; temporal positions; VisualVAE temporal chunking/blending and
ImageNet de-normalization; AudioVAE normalization, convolution, SnakeBeta,
clipping, and stereo order; or MP4 audio interleaving and A/V validation.

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
| `h3_euler_bf16` in `h3_shaders.metal` plus Diffusers Euler update | `sampling.hip` | Operation-boundary translation to one gfx1151 HIP kernel; F32 sample state and final-head velocities, Diffusers' materialized denoised/blend operation order, bounded extrapolation, bounds validation, and no CPU tensor fallback |
| `tests/test_h3.c`, `tests/test_bf16.c`, and pinned Diffusers scheduler | MiniMax H3 sampling host/HIP tests | Exact `torch.linspace` schedule/layout/row-map/noise hashes, temporal cases, overflow/malformed inputs, round trips, reuse masks, and byte-exact single-step plus complete short-trajectory device Euler coverage |

Current DiT-boundary adaptations:

| Upstream source | Local destination | Method and material changes |
| --- | --- | --- |
| `h3_dit.c` block load/run paths | `dit.hpp`, `dit.cpp`, `dit.hip` | One-block BF16 session with model-private tensor contracts, direct shard loading, atomics-disabled rocBLAS projections warmed before execution, fixed activation storage, explicit stream/cancellation/telemetry, and no allocation during block execution |
| DiT kernels in `h3_shaders.metal` | `dit_ops.cuh` | Operation-boundary HIP translation for casts, residual arithmetic, SiLU/SwiGLU, RMS/layer/AdaLN, gated residuals, grouped QKV interpretation, per-head Q/K norm, partial 3D MM-RoPE, and full SDPA |
| `tests/test_real_dit_block.c` | `minimax_h3_dit_hip_test.hip`, `tools/gufo/h3_dit_golden.py` | Odd-tail analytic fixtures plus an operator-owned 528-row block-0 oracle produced by direct ROCm PyTorch formulas; retained payloads are never committed |

Current full-denoiser adaptations:

| Upstream source | Local destination | Method and material changes |
| --- | --- | --- |
| `h3_dit_schedule.c` | `denoiser.hpp`, `denoiser.cpp`, `denoiser.hip` | Independent C++/HIP implementation of unique video/audio timestep rows, F32 timestep embedding, streamed per-block AdaLN projection, h3.c-compatible gate scoring/ranking with protected boundary blocks, final modulation, and explicit phase telemetry |
| token refinement, patch packing, core loop, and final heads in `h3_dit.c` | `denoiser.hip` | Complete text-only 50-block BF16 route; all core weights resident, device-resident hidden-state ping-pong on one ordered stream, fixed per-block scratch arenas, full attention, modality maps, F32 final heads and velocity boundaries, cancellation after every block/step, and no CPU tensor compute |
| GPU Euler denoise path in `h3_dit.c` plus Diffusers scheduler semantics | `denoiser.hip`, `sampling.cpp`, `sampling.hip` | Diffusers 5/20/50-point grids driving 4/19/49 evaluations, gate-ranked active blocks, whole-velocity reuse with two F32 boundaries, independent video/audio coefficients and two-stage operation order, bounded extrapolation, and device-resident F32 samples |
| `tests/test_real_dit.c`, `tests/test_semantic_dit.c` | `minimax_h3_denoiser_hip_test.hip`, `tools/gufo/h3_denoiser_golden.py` | Tiny lifecycle smoke and one independently generated 528-row full velocity forward; sampler trajectories remain covered by analytic host/HIP tests rather than routine multi-step model execution |

Current output-decoder adaptations:

| Upstream source | Local destination | Method and material changes |
| --- | --- | --- |
| VisualVAE load, temporal chunk, tiling, and selected-frame paths in `h3_video_vae.c` plus released Diffusers VisualVAE defaults | `video_vae.hpp`, `video_vae.cpp`, `video_vae.hip` | C++20/HIP phase with exact released F32 weights, fixed 256-pixel spatial tiles, seven-latent/22-frame chunks, bounded spatial/temporal output composition, selected-frame pruning, cancellation, and reusable scratch |
| VisualVAE kernels in `h3_shaders.metal` | `video_vae_ops.cuh` | Independent HIP implementation of latent normalization, patch embedding, RMS/layer norm, once-per-head QKV normalization/RoPE, stable F32 full attention, vectorized SwiGLU, residual scaling, output projection, and selected RGB unpack |
| `h3_audio_vae.c` decoder path | `audio_vae.hpp`, `audio_vae.cpp`, `audio_vae.hip` | Released F32 decoder only; exact tensor contracts, resident normalized convolution weights, stereo-as-batch execution, seven BigVGAN stages, cancellation, and memory/fault/swap telemetry; unused encoder attention is excluded |
| AudioVAE kernels in `h3_shaders.metal` | `audio_vae_ops.cuh` | Independent HIP implementation of latent normalization, weight normalization, bounded F32 im2col plus rocBLAS Conv1d/ConvTranspose1d, released two-stage alias-free SnakeBeta, residual accumulation, clipping, and channel-major PCM output |
| `h3_ffmpeg.c`, `tests/test_av_mux.c` | `media.hpp`, `media.cpp`, `minimax_h3_media_test.cpp` | C++20 two-pipe FFmpeg process with bounded PCM interleave staging, nonblocking cancellation-aware writes, deterministic cleanup, headless Nix dependency, FFprobe codec/geometry/rate/duration/sync validation, and no temporary uncompressed media file |
| `tests/test_real_video_vae.c`, `tests/test_real_audio_vae.c` | VisualVAE/AudioVAE HIP tests and direct PyTorch golden tools | Operator-owned frozen latent/frame/waveform/spectrogram payloads, analytic primitive fixtures, external hashes, one selected-frame video decode, one stereo audio decode, clipping, cancellation, and zero-swap gates |

Current text-only orchestration adaptations:

| Upstream source | Local destination | Method and material changes |
| --- | --- | --- |
| generation flow and exact/fast/aggressive examples in `h3.c`, `README.md` | `generation.hpp`, `generation.cpp`, `video_cli.hpp`, `video_cli.cpp` | Model-private C++ phase orchestrator and direct CLI with pinned exact/fast/aggressive/dev parameter contracts, prompt hashing, explicit overrides, selected-frame rapid iteration, progress/cancellation, atomic reports/outputs, and no implicit cache or model acquisition |
