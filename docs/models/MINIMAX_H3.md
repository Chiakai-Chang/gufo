# MiniMax H3

Gufo runs the text-only FL2VA path on Linux gfx1151: prompt encoding, the
50-block BF16 transformer, F32 Euler integration, VisualVAE, AudioVAE, and
MP4 composition. Model code lives in
[`src/models/minimax_h3`](../../src/models/minimax_h3).
First/last-frame conditioning, image references, Ref2VA, and 2K regeneration
are unsupported. Model tensor computation runs on HIP; host work handles
loading, planning, progress, and media composition.

## Model acquisition

| Artifact | Pinned identity |
| --- | --- |
| MiniMaxAI/MiniMax-H3, FL2VA | `42ed227ee7df40d41602854ae760620d6eb651fe` |
| Source manifest SHA-256 | `00a83367b87017f1f3d5547a963b20567a46d7aea44f0aac495c816147720d89` |
| MiniMax H3 Community License Agreement | August 2, 2026 |
| License SHA-256 | `59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44` |

The operator obtains the checkpoint and accepts its terms independently.
Gufo distributes engine code and binaries, without H3 weights, tokenizer
payloads, converted weights, or private authorization documents. Build,
installation, and server startup do not download this model.
See [third-party notices](../../THIRD_PARTY_NOTICES.md) for the model and
implementation licensing records.

After obtaining access, download the selected family and verify it:

```sh
nix develop -c hf download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --include LICENSE README.md model_index.json "FL2VA/**" \
  --local-dir /var/llms/huggingface/MiniMax-H3

nix develop -c python3 tools/h3/gufo-h3-manifest.py \
  --model-root /var/llms/huggingface/MiniMax-H3 \
  --verify src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json
```

The inventory tool checks revision metadata, payload hashes, safetensors
headers, tensor shapes/dtypes, and shard coverage without executing
model-provided Python. Runtime inspection validates the pinned inventory
before device allocation or job admission. Missing, partial, mismatched,
unsupported, or unsafe shard paths fail loading.

## CLI and server

Build with `nix build` and use `result/bin/gufo`. The presets are frozen in
[`generation.cpp`](../../src/models/minimax_h3/generation.cpp):

| Preset | Internal canvas | Output canvas | Sigma points / transitions | Active blocks | Velocity reuse interval |
| --- | --- | --- | ---: | ---: | ---: |
| `exact` | 512x512 | 512x512 | 50 / 49 | 50 | 1 |
| `exact-1344x768` | 1344x768 | 1344x768 | 50 / 49 | 50 | 1 |
| `fast` | 384x384 | 512x512 | 20 / 19 | 45 | 2 |
| `aggressive` | 320x320 | 512x512 | 20 / 19 | 40 | 3 |
| `dev` | 256x256 | selected 256x256 frames | 5 / 4 | 50 | 1 |

`exact` follows the released Diffusers schedule. Fast and aggressive are
approximate native presets with gate-ranked block thinning and extrapolated
velocity reuse; their schedule transitions do not all execute the transformer.
They protect blocks 0, 1, and 49 and execute retained blocks in model order.
Token reduction is disabled in every preset.

The CLI's `dev` preset uses the real 22-frame latent window, decodes frames
0, 11, and 21, and skips audio/muxing:

```sh
./result/bin/gufo video \
  --model /var/llms/huggingface/MiniMax-H3 \
  --preset dev --seed 42 --frames-dir /tmp/h3-preview --profile \
  "A red fox walking through snow"
```

For a complete user-requested MP4, select `exact` with
`--output /tmp/h3.mp4`, or `exact-1344x768` for the full-resolution preset.
`--latents-dir` optionally retains final F32 latents and provenance for an
offline comparison. `--attention-kernel scalar` is an explicit diagnostic;
the default is `row-parallel`.

Start a video-only server:

```sh
./result/bin/gufo serve \
  --host 127.0.0.1 --port 8080 \
  --video-model /var/llms/huggingface/MiniMax-H3 \
  --video-root /var/llms/huggingface/gufo-h3-jobs --video-ttl 3600
```

`POST /v1/videos` accepts JSON or multipart form data. For example:

```json
{
  "model": "minimax-h3",
  "prompt": "A red fox walking through snow",
  "size": "512x512",
  "seconds": 5,
  "gufo": {"seed": 42}
}
```

Requests use the released 24-fps, `17n + 5` frame alignment and 345-frame
ceiling. Durations in the 5–15-second range must align within that ceiling
(14.375 seconds of output); five seconds aligns to 124 frames. The
one-second/22-frame diagnostic extension is also supported. The prompt
encoder enforces **4,096 tokens after tokenization and NFC normalization**.
The HTTP byte budget is a separate request-resource limit.

The API's `dev` preset requires `size: "256x256"` and
`gufo.output_format: "ppm"` and returns one selected frame. An explicit
`gufo.frames` must match the aligned duration. Unsupported conditioning and
conflicting preset/model selections are rejected.

One worker executes H3 jobs, with one additional queued job; excess admissions
receive HTTP 429. Status, progress, cancellation, content ranges, expiry, and
restart recovery are described in [the server guide](../SERVER.md).
`DELETE /v1/videos/{id}` cancels work through every phase and reclaims its
artifacts. Parameter reports retain prompt hashes and generation controls;
persisted status omits prompt text and private paths.

## Execution and memory

The checkpoint's component inventory is larger than its simultaneous resident
set:

| Component inventory | Tensor bytes | GiB |
| --- | ---: | ---: |
| Qwen prompt encoder | 66,714,780,128 | 62.13 |
| AdaLN/time precompute | 26,142,079,488 | 24.35 |
| DiT core and heads | 40,138,350,656 | 37.38 |
| VisualVAE | 10,415,484,128 | 9.70 |
| AudioVAE | 605,306,340 | 0.56 |

The phase owner releases streams, mappings, HIP registrations, weights, and
arenas on completion, failure, or cancellation. Component inventories and
phase peaks must not be added as though all were resident together.

1. **Prompt encoding:** the private tokenizer uses NFC, byte BPE, and the
   checkpoint's added-token rules. It inserts no chat template or vision
   framing. The encoder captures the first 50 Qwen3-VL layers before the final
   norm. It releases the embedding after lookup and streams two layers at a
   time, overlapping transfer with compute.
2. **Denoiser setup:** timestep arithmetic stays F32 through SiLU. Each large
   AdaLN projection is loaded, used to precompute schedule constants, and
   released. Core loading is bounded to two workers. Construction validates
   BF16 GEMM plans and releases temporary validation/staging buffers.
3. **Denoising:** retained core blocks stay resident. Two BF16 device buffers
   carry hidden states on one ordered stream; per-block scratch is stable.
   F32 sample and velocity buffers implement Diffusers' two-stage Euler
   operation order. Reuse retains the previous video/audio velocities.
   Cancellation is checked at block and step boundaries.
4. **VisualVAE:** fixed 256-pixel spatial tiles and seven-latent/22-frame
   temporal chunks preserve decoder context. Selected-frame decoding skips
   unrelated chunks and readback while retaining required model work.
5. **AudioVAE and output:** F32 decoder convolution uses bounded im2col/GEMM;
   left/right channels remain independent. Unused audio-encoder weights are
   excluded. FFmpeg receives bounded RGB/PCM streams and produces H.264/AAC
   with 24-fps video and stereo 32-kHz audio.

Telemetry separates inspection, tokenization, loading, AdaLN precompute,
fresh denoiser forwards, sampler time, VAE work, media composition, faults,
swap, and memory peaks. `first_preview_ms` measures availability of the
requested frame set, not a streamed first frame.

## Quality and focused validation

The authoritative contract and component payload identities live in:

- [quality contract](../../tests/fixtures/minimax_h3/quality-contract-v1.json)
- [DiT block oracle](../../tests/fixtures/minimax_h3/dit-block0-oracle-v1.json)
- [denoiser oracle](../../tests/fixtures/minimax_h3/denoiser-oracle-v1.json)
- [VisualVAE oracle](../../tests/fixtures/minimax_h3/video-vae-oracle-v1.json)
- [AudioVAE oracle](../../tests/fixtures/minimax_h3/audio-vae-oracle-v1.json)

A changed prompt, seed, shape, schedule, block/reuse policy, metric, or ceiling
requires a new contract version. Never loosen a threshold to admit a
candidate. Numerical comparisons require matching shapes and zero non-finite
values.

| Boundary | Frozen strict ceiling |
| --- | --- |
| BF16 DiT block | relative max and relative L2 below `1e-2` |
| Qwen layer-50 output | relative max below `0.1`, relative L2 below `0.05` |
| 256x256x22 video velocity | relative max below `0.05`, relative L2 below `0.04` |
| 256x256x22 audio velocity | relative max and relative L2 below `0.05` |
| VisualVAE | relative max and relative L2 below `0.05` |
| AudioVAE waveform | max absolute below `1e-3`, relative L2 below `0.05` |
| AudioVAE STFT magnitudes | relative max below `0.1`, relative L2 below `0.08` |

Use the smallest independent oracle covering the change:

| Changed boundary | Validation |
| --- | --- |
| Loader/tokenizer/API | malformed inventory, exact token IDs, lifecycle and protocol tests |
| Sampler or schedule | analytic host/HIP layout, shifted grids, seeded noise, Euler trajectory, reuse ranking |
| DiT operator/block | analytic primitives and one frozen 528-row block; profile one 1,872-row block |
| Cross-block denoiser | one complete forward against the frozen conditioning and input latents |
| VisualVAE | one selected-frame tile against its F32 teacher |
| AudioVAE | one stereo waveform and deterministic STFT comparison |

Normal development does not run complete denoising schedules or generate full
videos. A selected-frame smoke may follow a change crossing component
boundaries; full generation and delivery-quality studies are explicit release
work. A playable MP4 or recognizable frame alone cannot qualify an
optimization.

Dependency-light schema/corruption checks and offline ML checks are separate:

```sh
nix build .#checks.x86_64-linux.h3-quality
nix build .#checks.x86_64-linux.h3-ml-quality

nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test --target minimax_h3_dit_hip_test
nix develop -c ctest --preset gpu-full \
  -R '^minimax_h3_dit_analytic$' --output-on-failure
```

Real-checkpoint tests use operator-owned artifacts and explicit
`GUFO_H3_MODEL_ROOT` plus component oracle paths. For a cross-block change:

```sh
GUFO_H3_MODEL_ROOT=/var/llms/huggingface/MiniMax-H3 \
GUFO_H3_DENOISER_GOLDEN=/var/llms/huggingface/gufo-h3-oracles/denoiser-256x256x22-forward-v2 \
  nix develop -c ./build/gpu-test/minimax_h3_denoiser_hip_test --oracle-forward
```

Teacher capture tools are model-scoped under `tools/gufo/h3_*_golden.py`
(prompt, DiT, denoiser, video VAE, and audio VAE). The prompt teacher loads
only the selected checkpoint layers in Transformers; component teachers
execute independent ROCm PyTorch formulas. The denoiser teacher was also
checked against the pinned converted Diffusers transformer. Identical
integer seeds do not equate Gufo's PCG/Box-Muller noise with PyTorch's RNG:
cross-runtime comparisons inject the same retained noise tensors.

External payloads stay outside Git in content-addressed directories. Each
manifest binds the contract, model/source and oracle revisions, dtypes,
conditioning hash, shapes, generation parameters, and every payload hash.
Paths, role coverage, metrics, and hashes are validated before comparison:

```sh
nix develop -c python3 tools/h3/gufo-h3-quality.py verify \
  --artifact /var/llms/h3-oracles/sha256/<sha256>
```

The following retained measurements document component evidence; this
documentation cleanup does not rerun or extend it:

| Boundary | Relative L2 | Relative max |
| --- | ---: | ---: |
| prompt layers 1 and 50, six-token fox prompt | `0` | `0` |
| 528-row block output | `0.00461029` | `0.00483092` |
| 512x512x22 refined text | `0.00469088` | `0.00167411` |
| 512x512x22 block-0 modulation | `0.0022895` | `0.00478469` |
| 512x512x22 video velocity | `0.0132707` | `0.0154553` |
| 512x512x22 audio velocity | `0.00731056` | `0.0149873` |
| selected VisualVAE frames | `8.14968e-7` | `2.24925e-6` |
| AudioVAE waveform | `1.08936e-5` | max absolute `6.03162e-5` |
| AudioVAE STFT | `3.62714e-6` | `5.8276e-6` |

Component agreement does not establish end-to-end semantic quality. Pre-audit
multi-step latents and checkerboard video captures are invalid promotion
evidence. Fast/aggressive delivery quality must be compared with the audited
exact route. Delivery reports bind sibling parameter documents and measure
latent error, per-frame/temporal SSIM and pinned LPIPS, stereo waveform/STFT
error, durations/A/V sync, and human prompt-adherence review.

## Profiling

Use release binaries and the focused tools in [the performance guide](../PERFORMANCE.md).
Retain the prompt/input hashes, seed, geometry, model/engine revisions,
operation contract, timings, memory, and swap alongside each result.
An independent numerical gate must pass before a speed result is retained.
Raw profiles and teacher payloads remain outside Git.

`tools/gufo/h3_profile.py` reports phase and complete-generation telemetry;
complete generation requires its explicit `--allow-full-generation` option.
`tools/gufo/h3_preset_quality.py` compares delivered media and frozen preset
provenance. `tools/gufo/h3_cache_control.py` performs an optional
file-cache eviction request without launching generation; its advisory result
does not prove a cold cache.

Retained experiments: device-resident block chaining, split QKV/AdaLN
normalization, F32 convolution GEMMs, fixed-shape attention, and VisualVAE
wave-local LayerNorm passed their component gates. Direct mapped execution
of repeatedly reused DiT weights regressed; alternate softmax reduction orders
failed quality. Full-resolution attention savings inferred from one block
remain extrapolations until a complete route is measured.

## Upstream provenance

### Native implementation reference

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

#### Audited implementation differences

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
| request duration | released pipeline accepts aligned 5--15 second requests | short native development examples | HTTP uses the same `17n + 5` alignment and 15-second ceiling at both output sizes; maximum 345 frames (14.375 seconds). The explicit one-second/22-frame diagnostic extension remains available |
| prompt length | pipeline/model tokenizer contract | implementation-specific | prompt encoder accepts up to 4,096 tokens after tokenization and normalization; HTTP/job admission does not equate UTF-8 bytes with tokens |
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

### TensorOps ancestry

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

### Adaptation record

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
| text kernels in `h3_shaders.metal` | `prompt_encoder_ops.hip.hpp` | Operation-boundary translation to HIP; BF16 rounding and FP32 reductions retained; no Metal/MPSGraph or ccv TensorOps code imported |
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
| DiT kernels in `h3_shaders.metal` | `dit_ops.hip.hpp` | Operation-boundary HIP translation for casts, residual arithmetic, SiLU/SwiGLU, RMS/layer/AdaLN, gated residuals, grouped QKV interpretation, per-head Q/K norm, partial 3D MM-RoPE, and full SDPA |
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
| VisualVAE kernels in `h3_shaders.metal` | `video_vae_ops.hip.hpp` | Independent HIP implementation of latent normalization, patch embedding, RMS/layer norm, once-per-head QKV normalization/RoPE, stable F32 full attention, vectorized SwiGLU, residual scaling, output projection, and selected RGB unpack |
| `h3_audio_vae.c` decoder path | `audio_vae.hpp`, `audio_vae.cpp`, `audio_vae.hip` | Released F32 decoder only; exact tensor contracts, resident normalized convolution weights, stereo-as-batch execution, seven BigVGAN stages, cancellation, and memory/fault/swap telemetry; unused encoder attention is excluded |
| AudioVAE kernels in `h3_shaders.metal` | `audio_vae_ops.hip.hpp` | Independent HIP implementation of latent normalization, weight normalization, bounded F32 im2col plus rocBLAS Conv1d/ConvTranspose1d, released two-stage alias-free SnakeBeta, residual accumulation, clipping, and channel-major PCM output |
| `h3_ffmpeg.c`, `tests/test_av_mux.c` | `media.hpp`, `media.cpp`, `minimax_h3_media_test.cpp` | C++20 two-pipe FFmpeg process with bounded PCM interleave staging, nonblocking cancellation-aware writes, deterministic cleanup, headless Nix dependency, FFprobe codec/geometry/rate/duration/sync validation, and no temporary uncompressed media file |
| `tests/test_real_video_vae.c`, `tests/test_real_audio_vae.c` | VisualVAE/AudioVAE HIP tests and direct PyTorch golden tools | Operator-owned frozen latent/frame/waveform/spectrogram payloads, analytic primitive fixtures, external hashes, one selected-frame video decode, one stereo audio decode, clipping, cancellation, and zero-swap gates |

Current text-only orchestration adaptations:

| Upstream source | Local destination | Method and material changes |
| --- | --- | --- |
| generation flow and exact/fast/aggressive examples in `h3.c`, `README.md` | `generation.hpp`, `generation.cpp`, `src/cli/video/video.cpp` | Model-private C++ phase orchestrator and direct CLI with pinned exact/fast/aggressive/dev parameter contracts, prompt hashing, explicit overrides, selected-frame rapid iteration, progress/cancellation, atomic reports/outputs, and no implicit cache or model acquisition |
