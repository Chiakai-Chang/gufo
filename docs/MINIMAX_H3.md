# MiniMax H3 Integration Boundary

Status: active engineering policy, 2026-08-22

## Purpose

This document defines how Strix-Halo.cpp may integrate an operator-supplied
MiniMax H3 checkpoint without distributing model weights or treating the
engine's MIT license as permission to use the model.

It is an engineering record, not legal advice.

## Pinned Inputs

The initial supported source is:

```text
repository: MiniMaxAI/MiniMax-H3
revision:   42ed227ee7df40d41602854ae760620d6eb651fe
task:       FL2VA
license:    MiniMax H3 Community License Agreement, August 2, 2026
license sha256:
  59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44
```

The upstream port used as an implementation reference is pinned separately by
the H3 source-manifest work. No model weights are copied from that project.

## Reproducible Acquisition and Verification

After independently obtaining access from MiniMax, a clean machine downloads
only the selected family:

```sh
hf download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --include LICENSE README.md model_index.json "FL2VA/**" \
  --local-dir /var/llms/huggingface/MiniMax-H3
```

The native inventory tool parses JSON and safetensors metadata directly. It
does not import or execute Python supplied by the model repository:

```sh
nix develop -c python3 tools/strix-h3-manifest.py \
  --model-root /var/llms/huggingface/MiniMax-H3 \
  --verify src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json
```

Verification checks the Hugging Face revision metadata for every file, hashes
every payload, validates every safetensors header/index/tensor, and rejects
missing or unreferenced shards, unsupported dtypes, Ref2VA, 2K regeneration,
pickle checkpoints, and wrong revisions before runtime device allocation.

Numerical and audiovisual work follows the frozen
[MiniMax H3 quality-oracle contract](MINIMAX_H3_QUALITY.md). Large teacher
outputs remain outside Git in schema-validated, content-addressed directories;
CI never regenerates them.

The model-private loader's phase boundaries and first gfx1151 residency
measurement are recorded in
[the MiniMax H3 residency baseline](../src/models/minimax_h3/RESIDENCY.md).

## Text-Only Prompt Encoder

The first executable H3 boundary is the FL2VA tokenizer plus the first 50
Qwen3-VL-32B decoder layers. It intentionally stops after layer index 49,
before the checkpoint's final text norm, because that BF16 hidden state is the
conditioning tensor consumed by H3.

The tokenizer is model-private. It implements the pinned NFC, GPT-2 byte
encoding, BPE merge order, added-token policies, longest/earliest special-token
matching, and empty-prompt pad behavior. ICU supplies complete Unicode NFC and
category handling. No BOS, EOS, chat template, or vision framing is inserted
for the initial text-only path.

The HIP encoder:

- validates the fixed 151936x5120 embedding and all 11 tensors in each layer;
- keeps activations and operation boundaries in BF16 with FP32 reductions and
  matrix accumulation;
- uses hipBLASLt BF16 projections plus model-private RMSNorm, Q/K norm, RoPE,
  causal GQA, residual, and SwiGLU kernels;
- loads only the embedding and layers 0 through 49;
- overlaps the next layer's read-only registered mapping/device copy with
  current-layer execution, then unregisters every host page;
- admits one prompt-encoder phase at a time;
- rejects vision spans and deepstack inputs before allocation;
- does not call the repository's Qwen language-model runtime and has no CPU
  tensor-compute fallback.

The offline teacher capture is reproducible but not part of the production
package:

```sh
nix develop -c python tools/strix/h3_prompt_golden.py \
  --model-root /var/llms/huggingface/MiniMax-H3 \
  --layers 50 \
  --output /var/llms/huggingface/strix-h3-oracles/\
fox-layer50-transformers.bf16
```

The raw BF16 oracle and metadata remain outside Git.

## Text-Only Layout and Sampler

The initial FL2VA sampler boundary is model-private and text-only. It accepts
checked 32-pixel canvas multiples within the released 768x1344 pixel-area
limit, aligns requests to `5 + 17*n` frames through the 362-frame ceiling, and
derives the separate video and audio latent timelines at 24 fps.

Before any large DiT allocation, the host planner freezes:

- text, target-audio, then target-video row order;
- three-dimensional MM-RoPE coordinates and per-step AdaLN row maps;
- 2x2 visual patch order and stereo audio row order;
- independent video/audio shifted sigma grids with terminal zero;
- independent PCG/Box-Muller generators initialized from the same request
  seed;
- denoiser evaluation selection and bounded linear velocity extrapolation.

This is host planning and input construction, not a CPU tensor-inference
fallback. Euler sample state remains F32 in gfx1151 device memory. The HIP
kernel reads the most recent and previous BF16 DiT velocities, applies the
frozen extrapolation ratio, and updates only the requested device range with
the same fused arithmetic as the pinned h3.c boundary.

First/last-frame conditioning and ordered Ref2VA inputs are deliberately not
accepted by this text-only layout API; their row kinds remain future work.

## BF16 DiT Block Baseline

The first transformer-core boundary is one complete dense H3 block at the
released width: hidden size 5376, 56 grouped QKV heads, head dimension 128,
attention width 7168, and SwiGLU width 14336.

`DitBlockSession` loads only the selected block's eight BF16 tensors directly
from the validated safetensors inventory. Session creation allocates every
input, retained boundary, activation, comparison buffer, and rocBLAS resource.
It pins and byte-validates the gfx1151 rocBLAS solution before admitting a
run. `Run` performs no device allocation and keeps a single explicit stream.

The exact path is:

1. RMS AdaLN slots 0/1;
2. BF16 grouped QKV projection;
3. per-head Q/K RMSNorm and 3D partial MM-RoPE;
4. deterministic full scaled dot-product attention;
5. BF16 output projection and gated residual slot 2;
6. RMS AdaLN slots 3/4;
7. BF16 FC1, SwiGLU, FC2, and gated residual slot 5.

Model-private utility kernels also cover BF16/F32 casts, add/subtract, SiLU,
layer norm, F32 patch projection to BF16, and BF16 final projection to F32.
The patch/output helpers retain straightforward fixed-order accumulation for
the initial quality route; later pipeline work may replace them only against
the same retained boundaries.

The independent oracle is generated outside production:

```sh
nix develop -c python3 tools/strix/h3_dit_golden.py \
  --model-root /var/llms/huggingface/MiniMax-H3 \
  --output /var/llms/huggingface/strix-h3-oracles/dit-block0-528-v1
```

Run the analytic and external-model gates:

```sh
./build-h3-173/minimax_h3_dit_hip_test --analytic

STRIX_H3_MODEL_ROOT=/var/llms/huggingface/MiniMax-H3 \
STRIX_H3_DIT_GOLDEN=/var/llms/huggingface/strix-h3-oracles/dit-block0-528-v1 \
  ./build-h3-173/minimax_h3_dit_hip_test --real
```

Sparse attention, token reduction, cross-block fusion, reuse, layer thinning,
and quantized DiT weights are not part of this baseline.

## Complete BF16 Transformer and Denoiser

`DenoiserSession` is the quality-first full FL2VA transformer route. Creation
consumes the six-row layer-50 prompt tensor, refines it through
`condition_proj` and both token-refiner blocks, precomputes the independent
video/audio timestep schedule, and then admits all 50 dense core blocks.

The memory lifecycle is explicit:

1. prompt-encoder ownership ends before denoiser creation;
2. timestep embedding weights are loaded, executed, and released;
3. each block's 96,768-wide AdaLN projection is loaded, executed for all
   unique timestep rows, copied into that block's persistent constants, and
   released before the next projection;
4. the corresponding core block and its pre-sized activation arena become
   resident;
5. the final AdaLN projection is released before patch and output heads are
   admitted.

Every core block uses the byte-validated BF16 projection policy from the
one-block baseline. The current quality route keeps a separate fixed arena per
block so setup can fail atomically and timed execution performs no device
allocation. Between block streams, BF16 residual rows cross a preallocated
host staging pair. This is synchronization and byte transport on the unified
memory machine, not CPU tensor compute; all projection, normalization,
attention, MLP, head, and Euler arithmetic remains on gfx1151.

A fresh forward:

- projects packed F32 audio/video latents to BF16 and packs refined text,
  audio, then video rows;
- executes all 50 blocks with full attention and the released modality row
  maps;
- applies final audio/video AdaLN and F32 output heads;
- retains F32 velocities for diagnostics and BF16 velocity boundaries for the
  device Euler update.

The quality baseline uses all 50 blocks and one fresh evaluation per schedule
step. The serving API also accepts an explicit active block prefix and
whole-denoiser reuse interval. Skipped evaluations reuse the latest two BF16
velocity boundaries with the frozen bounded extrapolation plan; the first and
last schedule steps are always evaluated. Sparse attention, token reduction,
core-residual reuse, and quantization remain disabled.

Generate the independent four-step 256x256 teacher outside the repository:

```sh
nix develop -c python3 tools/strix/h3_denoiser_golden.py \
  --model-root /var/llms/huggingface/MiniMax-H3 \
  --conditioning /var/llms/huggingface/strix-h3-oracles/\
fox-layer50-transformers.bf16 \
  --output /var/llms/huggingface/strix-h3-oracles/\
denoiser-256x256x22-4step-v1
```

Run the rapid 256x256 quality gate:

```sh
STRIX_H3_MODEL_ROOT=/var/llms/huggingface/MiniMax-H3 \
STRIX_H3_DENOISER_GOLDEN=/var/llms/huggingface/strix-h3-oracles/\
denoiser-256x256x22-4step-v1 \
  ./build-h3-173/minimax_h3_denoiser_hip_test --oracle
```

Production-shape development coverage uses the real 512x512, 22-frame,
1,872-row layout but only two denoising evaluations. This catches
shape-dependent attention, patch-layout, and sampler errors without spending
hours producing a delivery video:

```sh
nix develop -c python3 tools/strix/h3_denoiser_golden.py \
  --model-root /var/llms/huggingface/MiniMax-H3 \
  --conditioning /var/llms/huggingface/strix-h3-oracles/\
fox-layer50-transformers.bf16 \
  --output /var/llms/huggingface/strix-h3-oracles/\
denoiser-512x512x22-2step-v1 \
  --width 512 --height 512 --steps 2 --noise-mode seeded

STRIX_H3_MODEL_ROOT=/var/llms/huggingface/MiniMax-H3 \
STRIX_H3_DENOISER_GOLDEN_512=/var/llms/huggingface/strix-h3-oracles/\
denoiser-512x512x22-2step-v1 \
  ./build-h3-175/minimax_h3_denoiser_hip_test --oracle-512
```

There is deliberately no 50-step development test. The `exact` preset remains
available to end users and for rare, explicitly requested release validation,
but ordinary implementation, profiling, and CI use short component oracles.

## VisualVAE and AudioVAE Output Phases

The two released FL2VA output decoders are independent model-private phases.
They consume only the final normalized latents, never overlap their complete
weight inventories, and expose cancellation and memory telemetry separately
from the denoiser.

`VideoVaeDecoder` keeps the released F32 transformer arithmetic on gfx1151.
It chooses deterministic 256–320 pixel tiles with at least 64 pixels of
overlap, decodes legal seven-latent temporal chunks into 22-frame windows, and
blends the released five-latent/17-frame stride. The selected-frame API still
runs every complete model chunk needed by the requested global frame indexes;
it omits unrelated chunks and RGB reconstruction/readback only. Spatial and
temporal stitching are output composition, not CPU model inference.

Generate and run the operator-owned VisualVAE oracle:

```sh
nix develop -c python3 tools/strix/h3_video_vae_golden.py \
  --model-root /var/llms/huggingface/MiniMax-H3 \
  --latent /var/llms/huggingface/strix-h3-oracles/\
denoiser-256x256x22-4step-v1/video_final.f32 \
  --output /var/llms/huggingface/strix-h3-oracles/\
video-vae-256x256x22-v1

STRIX_H3_MODEL_ROOT=/var/llms/huggingface/MiniMax-H3 \
STRIX_H3_VIDEO_VAE_GOLDEN=/var/llms/huggingface/strix-h3-oracles/\
video-vae-256x256x22-v1 \
  ./build-h3-173/minimax_h3_video_vae_hip_test
```

`AudioVaeDecoder` accepts channel-major normalized F32 `[32,2,T]` audio
latents and returns channel-major F32 `[2,T*800]` PCM at 32 kHz. The native
route covers the released input projection, exact weight normalization,
Conv1d, ConvTranspose1d, seven BigVGAN upsampling stages, 21 residual blocks,
127 alias-free SnakeBeta activations, final clipping, and channel
recombination. Left and right channels remain independent batches throughout.
The unused audio encoder and its attention projection are not loaded into the
text-to-video serving path.

All 127 released alias-free upsample/downsample filters were checked
byte-for-byte and are identical, so the runtime shares one resident pair
without changing arithmetic. Model convolution and activation compute stays
on gfx1151; host work is limited to checkpoint I/O, progress, final PCM
readback, evaluation, and media output.

Generate and run the operator-owned AudioVAE oracle:

```sh
nix develop -c python3 tools/strix/h3_audio_vae_golden.py \
  --model-root /var/llms/huggingface/MiniMax-H3 \
  --latent /var/llms/huggingface/strix-h3-oracles/\
denoiser-256x256x22-4step-v1/audio_final.f32 \
  --output /var/llms/huggingface/strix-h3-oracles/\
audio-vae-256x256x22-v1

STRIX_H3_MODEL_ROOT=/var/llms/huggingface/MiniMax-H3 \
STRIX_H3_AUDIO_VAE_GOLDEN=/var/llms/huggingface/strix-h3-oracles/\
audio-vae-256x256x22-v1 \
  ./build-h3-173/minimax_h3_audio_vae_hip_test --real
```

The MP4 writer sends RGB24 and channel-major PCM to FFmpeg through two
concurrent nonblocking pipes. Audio interleaving uses a bounded 4,096-sample
chunk rather than a second full waveform. The child, both descriptors, and
partial output are reclaimed on success, write failure, child failure, or
cancellation. The pinned Nix package uses headless FFmpeg with H.264 and AAC;
FFprobe validates codec, dimensions, 24 fps, stereo 32 kHz audio, duration,
and start-time synchronization. When internal and output canvases differ,
FFmpeg performs an explicit Lanczos scale before H.264 encoding. Media tools
are runtime infrastructure only, not a CPU inference fallback.

## Text-to-Video CLI and Frozen Presets

`strix-server video` is the first complete text-only serving path. It requires
an explicit operator-supplied model directory, validates the pinned manifest,
tokenizes and encodes layer 50, denoises on gfx1151, decodes the released
VisualVAE and AudioVAE, and atomically publishes either an audiovisual MP4 or
selected diagnostic PPM frames. It never downloads weights and has no CPU
model-inference fallback.

The named preset contract is versioned as
`strix.minimax-h3-text-generation.v1`:

| Preset | Internal canvas | Output canvas | Steps | Active blocks | Denoiser reuse |
| --- | ---: | ---: | ---: | ---: | ---: |
| `exact` | 512x512 | 512x512 | 50 | 50 | 1 |
| `fast` | 384x384 | 512x512 | 20 | 45 | 2 |
| `aggressive` | 320x320 | 512x512 | 20 | 40 | 3 |
| `dev` | 256x256 | selected 256x256 frames | 4 | 50 | 1 |

Token reduction is false in every preset. The development preset decodes only
frames 0, 11, and 21 and skips AudioVAE and muxing, so numerical work can
iterate without waiting for a complete video.

Rapid development example:

```sh
nix develop -c ./build-h3-173/strix-server video \
  --model /var/llms/huggingface/MiniMax-H3 \
  --preset dev \
  --frames-dir /tmp/h3-fox-dev \
  --profile \
  "A red fox walking through snow"
```

Manual end-user or release exact output:

```sh
nix develop -c ./build-h3-173/strix-server video \
  --model /var/llms/huggingface/MiniMax-H3 \
  --preset exact \
  --seed 42 \
  --output /tmp/h3-fox-exact.mp4 \
  --latents-dir /tmp/h3-fox-exact-latents \
  --profile \
  "A red fox walking through snow"
```

Every admitted request writes a parameter document containing the preset,
prompt hash, seed, geometry, step/block/reuse values, and selected frames.
Successful runs add a timing, memory, denoiser-evaluation-count, and swap
report. Neither document contains the prompt text or model path. `SIGINT`
propagates through the phase cancellation token; temporary frames and media
are never published as final output.

Advanced geometry, step, block, reuse, output-canvas, and selected-frame
overrides exist for controlled experiments and are fully represented in the
parameter document. `--latents-dir` atomically retains final F32 video/audio
latents plus their shapes and hashes for an explicit teacher comparison. Its
manifest also binds the model/reference revisions, seed, prompt hash, exact
layer-50 conditioning hash, geometry, evaluation/block/reuse policy, and
attention kernel. The comparison rejects provenance drift before applying a
numeric tolerance. This diagnostic output is never enabled by the API.
`--attention-kernel scalar` retains the original deterministic SDPA as an
explicit profiling/rollback route; `row-parallel` is the named-preset default,
and the selected policy is included in parameters and telemetry. First-frame,
last-frame, and ordered-reference inputs are rejected until their future
conditioning milestones land. No implicit conditioning/model cache is enabled
in this CLI.

Sparse attention is explicitly future work. It must not change these preset
names or silently enter the exact route.

## Asynchronous Video API

`strix-server serve --video-model <DIR>` enables the versioned
`strix.video-api.v1` text-to-video API. When `--model` is not also supplied,
the process is video-only and does not load an unrelated text model:

```sh
nix develop -c ./build-h3-173/strix-server serve \
  --host 127.0.0.1 \
  --port 8080 \
  --video-model /var/llms/huggingface/MiniMax-H3 \
  --video-root /var/llms/huggingface/strix-h3-jobs \
  --video-ttl 3600
```

Startup inspects the operator-supplied directory against the pinned source
manifest before binding the listening socket. A missing, partial, or
incompatible checkpoint therefore fails configuration rather than creating a
job that is known to be unusable. There is no CPU inference fallback.

The supported routes are:

| Method | Path | Result |
| --- | --- | --- |
| `POST` | `/v1/videos` | Admit one asynchronous text-to-video job |
| `GET` | `/v1/videos/{id}` | Read stable status, progress, and error fields |
| `GET` | `/v1/videos/{id}/content` | Read completed MP4 or diagnostic PPM content |
| `DELETE` | `/v1/videos/{id}` | Cancel or remove a job and reclaim its artifacts |

`GET /v1/models` advertises `minimax-h3` and the
`minimax-h3-{exact,fast,aggressive,dev}` aliases. An alias selects its matching
frozen preset; if `strix.preset` is also present, the two must agree.

Create accepts `application/json` and the `multipart/form-data` shape used by
OpenAI video clients. JSON remains convenient for the nested Strix extension:

```json
{
  "model": "minimax-h3-fast",
  "prompt": "A red fox walking through snow",
  "size": "512x512",
  "seconds": "1",
  "strix": {
    "seed": 42,
    "output_format": "mp4"
  }
}
```

The equivalent standard text-only form selects the mode through the model
alias:

```sh
curl http://127.0.0.1:8080/v1/videos \
  -F model=minimax-h3-fast \
  -F 'prompt=A red fox walking through snow' \
  -F size=512x512 \
  -F seconds=1
```

Multipart duplicate fields and malformed boundaries fail closed. A multipart
`input_reference` file is recognized but rejected explicitly until the
first/last-frame conditioning milestone lands.

Exact, fast, and aggressive deliberately support one-second, 22-frame,
512x512 audiovisual MP4 files only. The development alias requires
`size: "256x256"` and `output_format: "ppm"`; it accepts a released
VisualVAE-decodable frame count through `strix.frames` (`22, 39, ... 362`) and
exactly one diagnostic frame through `strix.selected_frame`. Although the
denoiser geometry can represent five frames, the released VisualVAE's minimum
valid temporal chunk is seven latents and 22 output frames. The development
path therefore preserves the real 22-frame latent window while decoding and
reading back only the requested preview frame. First-frame, last-frame,
image-reference, and ordered-reference fields fail closed until their
conditioning milestones land.

One worker owns the H3 generation path and one additional request may wait in
the bounded queue. Further admissions return HTTP 429 with `Retry-After`.
Deleting the active job propagates cancellation through prompt encoding,
denoising, both VAEs, and media publication. Completed content supports a
single standard HTTP byte range and reports `Accept-Ranges: bytes`.

Job status is persisted atomically. Completed artifacts survive process
restart until their configured TTL; queued, interrupted, expired, deleted,
and partial artifacts are reclaimed. Status documents never store the prompt
or private filesystem paths. Parameter reports store only a SHA-256 prompt
digest plus the reproducibility controls, and telemetry contains bounded
numeric execution data.

## Full-Route Profiling

`tools/strix/h3_profile.py` drives the same public CLI route used by serving.
It is a manual release tool, not a development or CI gate. Development
optimization uses one independently frozen 528-row DiT block oracle for
correctness and one 1,872-row DiT block for performance and profiling. Neither
iteration gate runs all 50 blocks or advances a denoising schedule. A complete
50-step generation is run only when explicitly requested after those gates
pass.

The issue #175 performance workload is a manual single-block profile, not a
CTest target:

```sh
STRIX_H3_MODEL_ROOT=/var/llms/huggingface/MiniMax-H3 \
  ./build/hardware-test/minimax_h3_dit_hip_test --profile-1872
```

It executes block 0 once with deterministic production-shape inputs. The
original custom full-attention path and the retained batched-QK/softmax/PV
path measured:

| Attention implementation | Block GPU time |
| --- | ---: |
| custom row-parallel baseline | 38.518 s |
| batched rocBLAS QK/PV + parallel BF16 softmax | 1.076 s |

The retained implementation is `35.79x` faster for that block, a 97.2%
latency reduction. It allocates one shared F32 score matrix and one shared
BF16 probability matrix for the denoiser, 1.097 GiB at 1,872 rows; all 50
blocks reuse the same workspace rather than allocating per-block copies. The
independent 528-row block oracle remains the quality gate and stays below its
frozen 1% relative-error ceilings.

The retained profile attributes 435.7 ms to parallel softmax, 390.3 ms to
grouped QKV normalization/RoPE, and 208.9 ms to rocBLAS GEMMs. These are the
next measured optimization candidates; no full denoiser run is needed to rank
them.

Historical scalar phase timing, used only for bottleneck ranking because its
delivered MP4 is invalid quality evidence, attributes 98.326% of complete
latency to the denoiser, 1.240% to VisualVAE, 0.163% to prompt encoding, and
0.016% to AudioVAE. The next optimization priority therefore remains the DiT
forward. Dense projections/MLP and the host-staged block boundaries require
focused profiling before any further change; VisualVAE is a distant second.

The harness schedules each requested preset once, rejects `--rounds` values
other than one, and requires an acknowledgement before it can launch a
complete generation:

```sh
nix develop -c python3 tools/strix/h3_profile.py \
  --binary ./build-h3-173/strix-server \
  --model /var/llms/huggingface/MiniMax-H3 \
  --output-root /var/llms/huggingface/strix-h3-profiles/bf16-v1 \
  --presets exact \
  --rounds 1 \
  --cooldown-seconds 0 \
  --allow-full-generation
```

If a rare release comparison is requested, run the same command once for the
candidate with its distinct binary and output root. Both observations use the
same prompt, seed, model/reference revisions, and harness. No warm, repeated,
or eviction-requested full video is added.

The generated report combines the versioned parameter and runtime telemetry
with wall time, process read/write throughput, peak total/anonymous/file-backed/
shared-memory RSS, HWM/swap, and available AMDGPU power and selected-clock
samples. Raw profiler captures may retain an available temperature sensor, but
the comparison tool deliberately ignores it because ambient conditions differ
between operators. It records the binary hash, profiling-harness hash,
and kernel release while omitting prompt text, model paths, hostname, and the
private command line. Successful comparison runs require complete lowercase
SHA-256 identities for the binary, harness, prompt, and delivered output;
malformed or abbreviated identity strings fail closed.
The retained pre-selector scalar binary is identified by its pinned SHA-256.
Its older parameter document may omit `attention_kernel`; report and delivery
quality tools infer `scalar` only when the sibling profile binds that exact
binary hash, parameter document, and delivered output hash. Unknown binaries
with missing kernel provenance fail closed.
Timestamped raw monitor samples are retained alongside the aggregate so later
analysis can distinguish sustained behavior from a transient peak.
Every admitted release-comparison run must contain the complete hard-gate
telemetry set. The candidate's direct complete-generation latency, memory
peaks, loaded-clock coverage, and swap are compared with the single scalar
observation; no median or other repeated-run statistic is claimed.
Power and integrated energy are diagnostics rather than efficiency ceilings.
An optimized route may use the configured 120--140 W graphics envelope when
that power performs useful work and reduces complete-generation latency.
Sustained-clock throttling, swap, output quality, and evidence of busy-waiting,
redundant transfer, or recomputation remain promotion gates. Temperature is
neither required telemetry nor a comparison objective.

`tools/strix/h3_cache_control.py` remains available for a separately requested
cache diagnostic:

```sh
nix develop -c python3 tools/strix/h3_cache_control.py \
  --model /var/llms/huggingface/MiniMax-H3 \
  --output /var/llms/huggingface/strix-h3-profiles/cache-control.json
```

The cache-control artifact records files/bytes attempted, errors, memory
snapshots, and elapsed time without private filenames. Each regular target is
opened with `O_NOFOLLOW` and sized through the opened descriptor before the
advisory request. Linux `POSIX_FADV_DONTNEED` is advisory, so the result is
named `eviction-requested`, not asserted to be a physically cold cache. This
utility does not trigger a generation.

For a separately requested fast/aggressive delivery-quality study, compare a
delivered preset MP4 with its matching exact output:

```sh
nix develop -c python3 tools/strix/h3_preset_quality.py \
  --reference /var/llms/huggingface/strix-h3-profiles/bf16-v2/000-exact/output.mp4 \
  --candidate /var/llms/huggingface/strix-h3-profiles/bf16-v2/001-fast/output.mp4 \
  --candidate-label fast \
  --output /var/llms/huggingface/strix-h3-profiles/bf16-v2/001-fast/quality.json
```

The versioned report hashes both MP4s and compares all decoded RGB frames,
first/middle/last frames, adjacent-frame deltas, stereo waveform, deterministic
spectrogram, frame/audio durations, and A/V duration delta. It contains no
prompt, model path, hostname, or private command line. Promotion reports
require both profile-style `parameters.json` siblings, embed and hash those
privacy-safe documents, and reject mismatched provenance or preset contracts.
Missing parameter documents are accepted only with the explicitly diagnostic
`--allow-missing-parameters` switch. These delivery-level metrics supplement
rather than replace the retained component oracles.

The runtime telemetry separates model inspection, tokenizer work, prompt
weight loading/prefetch wait/submission/GPU time, AdaLN precompute, core load,
every fresh denoiser forward, sampler time, VisualVAE tiles, AudioVAE stages,
media composition, VisualVAE frame-set availability (`first_preview_ms`),
dispatch counts, page faults, swap, and phase memory peaks. That field is
recorded when the requested selected/full frame set returns; it is not a
streaming-first-frame claim. The single-observation comparison also derives
denoiser non-forward time as denoiser wall time minus the sum of fresh
forwards, so old and new binaries remain comparable even when a raw sampler
field changes granularity. Both issue #175 generations use the default
`first-observation` label rather than making an unsupported cache-state claim.

During development, a kernel or residency candidate is retained only when it
passes the independent single-block oracle, the frozen preset contracts, and
a single production-shape block profile showing useful work. A short
multi-step latent gate is reserved for integration changes that cross the
block, sampler, or schedule boundary; it is not rerun for an attention-kernel
iteration. Fast and aggressive do not add full benchmark videos.
Byte-identical complete MP4 comparison is reserved for an explicitly requested
release validation, not every implementation iteration.

## Operator Attestation

The operator of the dedicated Strix Halo development machine has stated that
the required MiniMax H3 terms or authorization have been obtained and accepted
for this work. The checkpoint is retained only on that operator-controlled
machine.

Strix-Halo.cpp does not attempt to adjudicate the operator's authorization,
store private license documents, or transmit credentials. Each downstream
operator must independently obtain the checkpoint from MiniMax and satisfy the
terms applicable to their territory, deployment, users, and outputs.

## Bring-Your-Own-Weights Contract

- The repository and its release artifacts contain no MiniMax H3 weights.
- Build, installation, and normal server startup never download H3.
- The operator supplies an explicit local model directory.
- The loader validates the pinned manifest, shard inventory, sizes, hashes,
  configurations, and tensor metadata before exposing the model.
- Missing, partial, mismatched, or unsupported files fail closed.
- The runtime does not search public mirrors or silently select another H3
  revision.
- Derived quantized weights remain local and are not publication artifacts.

The official Hugging Face acceptance flow or a separate MiniMax authorization
is the source of access. A Strix command-line flag or configuration entry must
never pretend to replace that external process.

## Distribution and Serving Boundary

Strix-Halo.cpp distributes numerical engine source and binaries only. It does
not distribute the H3 checkpoint, tokenizer, converted weights, or a hosted H3
service.

An operator who exposes H3 through the asynchronous video API owns the
deployment obligations imposed by the applicable MiniMax terms, including as
relevant:

- binding users to required restrictions;
- acceptable-use and safeguard controls;
- reporting and abuse-handling mechanisms;
- generated-content disclosures and attribution;
- territorial and commercial-use conditions;
- retention and removal procedures after authorization ends.

The server documentation must identify the selected model as MiniMax H3 and
must not imply that the engine authors sublicense the model.

## Release Gate

A source or binary release fails the H3 boundary when any of the following is
true:

- a model weight, tokenizer payload, generated quantized artifact, or private
  authorization file is included;
- startup or build logic automatically downloads the H3 checkpoint;
- the pinned source revision or public license metadata is absent;
- H3 can start from a partial or unvalidated local directory;
- server documentation describes the engine license as covering H3;
- a packaged example points users to an unofficial weight mirror.

Tracked test fixtures may contain only small synthetic tensors generated by
the project. They must not contain extracted checkpoint values.

## Required Runtime Behavior

The H3 implementation issues must preserve this boundary:

1. The loader consumes only an explicit operator-supplied path.
2. Artifact validation runs before device allocation or request admission.
3. API model discovery distinguishes engine support from local model
   availability.
4. A missing H3 checkpoint returns a deterministic configuration error.
5. Logs may report public repository/revision metadata but never access tokens,
   credentials, or private authorization material.
