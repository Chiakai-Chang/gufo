# MiniMax H3 Quality Oracles

Status: frozen foundation contract, 2026-08-22

## Rule

MiniMax H3 work is promoted in this order:

1. an independent analytic or pinned teacher oracle;
2. correctness and quality gates;
3. profiling;
4. optimization or quantization.

A successful kernel launch, finite latent, recognizable frame, or playable MP4
is not a quality result. The released exact route—BF16/F32 for the denoiser and
released F32 for both VAEs—is the reference for every later fast or quantized
route.

The dependency-light oracle and corruption tests run separately from the
pinned PyTorch teacher and LPIPS construction test:

```sh
nix build .#checks.x86_64-linux.h3-quality
nix build .#checks.x86_64-linux.h3-ml-quality
```

The generic host CTest suite does not import PyTorch. The dependency-light
`h3-quality` derivation is part of the canonical hosted PR gate. The
multi-gigabyte ROCm PyTorch/LPIPS closure is an explicit offline gate so a
GitHub-hosted runner does not exhaust its disk while realizing dependencies.

## Pinned Contract

The committed contract is
`tests/fixtures/minimax_h3/quality-contract-v1.json`. It pins:

- MiniMax H3 FL2VA revision
  `42ed227ee7df40d41602854ae760620d6eb651fe`;
- `h3.c` revision `8974cc055ea9c02fcd14cc27dfda3e1027c05153`;
- the source-manifest SHA-256;
- exact tokenizer IDs and special-token IDs from the operator-supplied
  checkpoint;
- a rapid 256x256x22 four-evaluation fox case;
- a retained 512x512x22 50-evaluation fox case;
- an independent 512x512x22 50-evaluation ceramic-mug case;
- component parity ceilings inherited from the pinned upstream tests.

Changing a prompt, seed, canvas, frame count, schedule, block count, reuse
interval, metric, or ceiling creates a new contract version. A threshold is
never loosened to admit a candidate.

## Oracle Pyramid

Small host fixtures cover token IDs, special tokens, temporal alignment,
packed-row maps, schedules, patching, seeded noise, reuse selection, and Euler
updates. Those values are hand-checkable and committed.

Component artifacts then retain:

- Qwen layer-50 hidden states;
- one full DiT block and AdaLN modulation;
- complete video and audio velocity outputs;
- VisualVAE latent-to-frame output;
- AudioVAE latent-to-stereo-waveform output.

End-to-end artifacts retain initial and final latents, selected or complete
frames, stereo audio, metrics, and human review. The rapid case decodes only
frames 0, 11, and 21. H3 is still run with its legal 22-frame latent geometry;
the selected-frame path is a development diagnostic, not a one-frame model.

The large artifacts are deliberately absent from Git. CI validates the schema,
metrics, corruption behavior, and committed small fixtures but never
regenerates teacher data.

## Content-Addressed Artifact Format

Every external artifact lives at:

```text
<store>/<sha256>/
  manifest.json
  ... payloads named by manifest entries ...
```

The directory name is the SHA-256 of the canonical manifest after omitting its
self-referential `artifact_id`. The manifest records:

- contract ID and file hash;
- model/source manifest and immutable revisions;
- oracle implementation and revision;
- runtime version, platform, hardware, command, storage/compute/accumulation
  dtypes;
- prompt and prompt hash, seed, canvas, frames, evaluations, blocks, and reuse;
- every payload's relative path, role, size, SHA-256, dtype, and shape;
- measured metrics and their frozen threshold source;
- at least two byte-identical captures.

Absolute paths, parent traversal, symbolic links, unreferenced files,
non-finite metrics, wrong hashes, and incomplete role sets fail closed.

Seal a completed staging directory atomically:

```sh
nix develop -c python3 tools/strix-h3-quality.py seal \
  --staging /var/llms/h3-oracles/staging/fox-layer50 \
  --template /var/llms/h3-oracles/specs/fox-layer50.json \
  --store /var/llms/h3-oracles/sha256
```

Verify without executing the model:

```sh
nix develop -c python3 tools/strix-h3-quality.py verify \
  --artifact /var/llms/h3-oracles/sha256/<sha256>
```

The production runtime never imports model-provided Python. A separately
isolated offline teacher capture may use a pinned official runtime, but that
runtime, its command, and its dependency versions become part of the artifact.

## Prompt-Encoder Frozen Evidence

The text-only port was checked against isolated Transformers 5.14.1 and ROCm
PyTorch 2.12.0/HIP 7.2.53211 on the supported Radeon 8060S. The teacher
instantiates only the Qwen3-VL text model on the meta device, loads the exact
embedding and requested layers from the pinned H3 shards, uses eager
attention, and captures the BF16 tensor entering the final RMSNorm.

For `A red fox walking through snow`, token IDs are
`32 2518 38835 11435 1526 11794`. External raw-BF16 payloads are:

| Boundary | Shape | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| after layer 1 | 6x5120 | 61,440 | `72c28520b592a79bf278fcced10f215b44e12864e803a7ca2a18e7528305ef08` |
| after layer 50 | 6x5120 | 61,440 | `8015af1d2a551a0509bccc84372a30087f66721613f909c3f99e36ad7e269692` |

The gfx1151 HIP implementation matched both payloads byte-for-byte: measured
relative max and relative L2 were zero. A repeated complete run produced the
same output bytes, 850 dispatches, and zero live registered host bytes.

## Host Sampler Frozen Evidence

The text-only sampler tests freeze the exact released linear base grid after
independent video shift 12 and audio shift 3. Concatenated little-endian F32
video/audio arrays have these SHA-256 values:

| Evaluations | Schedule SHA-256 |
| ---: | --- |
| 4 | `dee27052a1aa4aa508a7b81bbe39e6bb4b880dd2d61dd3953380c5d0da3eb686` |
| 7 | `cbd915b9d5fb117e3c35fb1e02771e0fdbe427736726dff0086de229ea01ffc0` |
| 20 | `f78a6b940322389481be761fb2dc503e869573f37c87c75e32e418e42b3b3e39` |
| 50 | `711fc5640698dec50ad677b604bd1cc95dcc3def734e0e7f88967c1d28ca2033` |

For six text rows and the 22-frame temporal shape, canonical packed-layout and
step-7 modulation-map hashes are:

| Canvas | Rows | Layout SHA-256 | Row-map SHA-256 |
| --- | ---: | --- | --- |
| 256x256 | 528 | `61be3a596b2670766a7e86b5a526e13438e89a68523d99eb5bdd9f2c8ca7f61a` | `8574df4af41e95bb6ec35f3ec9459ceb88a319f68ad07e9ec70af5a84cb6a6e3` |
| 512x512 | 1,872 | `c3481ff47fabdf534823889a95df5655bd19ac11ff1755f5daa7b6e1655d00b0` | `c95bbf9938ac6b5c2d901af8466d374f6e173953954872e19ec12ed82fec98a1` |

The 256x256x22 seed-42 initial video and audio noise payloads hash together to
`0b9e324f731605e8b6050b2c7cdc46a320b75609c426ea548adb35332204620d`.
Separate same-seed generators make the complete audio payload byte-identical
to the corresponding video prefix.

The gfx1151 analytic test keeps the sample in device F32 and matches the
pinned four-element BF16-velocity Euler result byte-for-byte on repeated
allocations. Zero-delta and out-of-range updates fail before launch.

## DiT Block Frozen Evidence

The block-0 teacher uses direct ROCm PyTorch formulas rather than the native
runtime. It loads only the eight released BF16 block tensors, constructs the
canonical six-text-row 256x256x22 layout, executes explicit FP32 full
attention, and retains every required BF16 boundary. The payloads remain
operator-owned; their immutable sizes and hashes are frozen in
`tests/fixtures/minimax_h3/dit-block0-oracle-v1.json`.

On the supported gfx1151 route, the complete 528-row block measured:

| Retained boundary | Relative L2 | Relative max |
| --- | ---: | ---: |
| attention AdaLN | `1.54491e-5` | `0.0015625` |
| attention output | `0.00186781` | `0.00414938` |
| MLP AdaLN | `0.00219937` | `0.00621118` |
| block output | `0.00476406` | `0.00956938` |

Every value is below the frozen `1e-2` limits. The session rejects a
pre-cancelled block, out-of-range row maps, changed tensor shapes, non-finite
retained values, and a rocBLAS solution that fails its construction-time
validation.

Operator-local rocprof/ISA records identify the actual full-attention,
grouped-QKV/RoPE, AdaLN, gate, SwiGLU, and gfx1151 rocBLAS kernels. Generated
profiles remain outside Git; only their reviewed conclusions and immutable
hashes belong in documentation or issue comments.

## Full Denoiser Frozen Evidence

The full-denoiser teacher is a direct ROCm PyTorch program. It does not call
the native runtime. It independently executes condition projection, two text
refiner blocks, F32 timestep embedding, all 50 released AdaLN projections,
all 50 dense transformer blocks, modality-specific final heads, and four
independent-schedule Euler updates.

The raw 256x256x22 payloads remain operator-owned. Their shapes, hashes,
arithmetic contract, and gates are frozen in
`tests/fixtures/minimax_h3/denoiser-oracle-v1.json`.

Measured gfx1151 parity was:

| Boundary | Relative L2 | Relative max | Frozen ceiling |
| --- | ---: | ---: | ---: |
| refined text | `0.00477293` | `0.00167411` | `0.02` |
| block-0 AdaLN projection | `0.00228314` | `0.00478469` | `0.02` |
| step-0 video velocity | `0.00497814` | `0.00734305` | `0.04` |
| step-0 audio velocity | `0.019351` | `0.0473417` | `0.05` |
| four-step video latent | `0.00447156` | `0.00575856` | `0.08` |
| four-step audio latent | `0.00927884` | `0.0210095` | `0.08` |

The first complete 528-row forward took 17.7 seconds with the deliberately
scalar, deterministic full-attention baseline. Four denoising evaluations
took 67.4 seconds. Peak accounted live memory was 42.41 GiB with zero process
swap. These numbers identify the unoptimized exact route; they are not a
performance target.

Two repeated step-0 forwards produced byte-identical complete velocity arrays.
The smoke gate also cancels before a forward and immediately after an Euler
step, then reuses the same fully resident session successfully. It finally
destroys and reloads the complete session and requires the fresh session to
reproduce the original velocity bytes. Cancellation is checked before every
block and after every sampler step.

### Retained 512x512 exact denoiser

The retained production-shape gate completed all 50 blocks and all 50 fresh
evaluations for 512x512x22. It measured 8.630 seconds of AdaLN precompute,
18.696 seconds of core load, 12,496.600 seconds of denoising, a 59.67 GiB
accounted peak, and zero swap. Repeated final latent payloads were
byte-identical:

| Payload | SHA-256 |
| --- | --- |
| final video latent | `0ebaf61b03ba48a5107ffce4ec04f7955e5c4966c9b85792d50acb9c09f8614b` |
| final audio latent | `e1da6b038d0f664a250d600e6aca48e3b227dc2f1006256adf8bcac784e75921` |

The separate reuse smoke ran five schedule positions with interval three,
performed exactly three fresh evaluations, reused the resident session, and
reproduced its repeated output bytes with zero swap.

## VisualVAE Frozen Evidence

The isolated F32 teacher retained the normalized latent, all 22 RGB frames,
and frames 0, 5, 11, 16, and 21:

| Payload | Bytes | SHA-256 |
| --- | ---: | --- |
| normalized latent | 172,032 | `23411f438f351f96632884476def1f17513a6d2105b42ebf698a6d69d79d52` |
| full frames | 17,301,504 | `36dd863ffcdf34c2f882f57f401d2c66d7fbc441f5185d32aa1b03db46ca24c1` |
| selected frames | 3,932,160 | `a6cc2b34da6b86f70d91dee8e152051ddc0bac0dc7cd5e1f8e605e7f37696ec7` |

The gfx1151 decoder measured full-frame relative L2 `3.51369e-7` and relative
max `9.93023e-7`; selected-frame relative L2 was `3.72615e-7` and relative max
`9.74843e-7`. Full versus selected output was exactly equal, and a second
selected decode was byte-identical. The final deterministic implementation
uses the actual gfx1151 wave32 width for normalization and attention while
each lane covers two dimensions of the released 64-wide head. It measured a
9.38 GiB phase peak and zero swap.

## AudioVAE Frozen Evidence

The isolated teacher retained the normalized latent, stereo waveform, and
deterministic STFT magnitudes:

| Payload | Shape | SHA-256 |
| --- | --- | --- |
| normalized latent | `[32,2,37]` | `f742e12e3f55d91886a1b4218e6e5a3e64efcd65bce63de6a6d42976f3b2e04c` |
| waveform | `[2,29600]` | `ea122eb789d73424ff1c3bbe89719963dd407377165259a4f861f5dba55f837c` |
| spectrogram | `[2,513,116]` | `20f3d1ca41c3b2a005654aec1c25313de6f849b7dbb7bf7b1ae5ffb92e100424` |

The native decoder measured waveform maximum absolute error `5.93364e-5`,
waveform relative L2 `1.10334e-5`, spectrogram relative L2 `3.84286e-6`, and
spectrogram relative max `5.21096e-6`. Decode took 1.900 seconds with a
0.251 GiB phase peak. Repeatability, phase residency, channel independence,
and cancellation gates passed.

## Rapid End-to-End Evidence

Two independent development-preset runs used the real 256x256x22 latent
window, four fresh evaluations, all 50 blocks, and selected-frame decode. They
completed in 159.4 and 163.7 seconds with zero swap. The three PPM files were
byte-identical between runs:

| Frame | SHA-256 |
| ---: | --- |
| 0 | `d7dcba1ad32ddf2cc08193a0738b0a7eb076b6a3206e133547b1cdb03efd0d35` |
| 11 | `2f057f85092cd4f6b2549e7960c8c5faf06f5023434c3289384a6e1dfc0276c9` |
| 21 | `3c23a10b4cad0c2ae2c51bc6b8f74fcd690c88f1338df0ba0ad9fabec7a9368d` |

A real asynchronous API job returned frame 11 with the same hash, proving
direct CLI/server parity. Create, progress, full content, byte range, persisted
privacy, restart recovery, delete, and artifact reclamation passed. A
deterministic tiny media integration run produced byte-identical H.264/AAC
MP4s; video was 24 fps, audio was stereo 32 kHz, and decoded duration differed
by 11.3 milliseconds. Those two development-preset runs were a one-time
determinism investigation, not a repetition policy. Current development uses
one invocation of each needed short oracle and does not rerun a passing full
route.

## Numerical Gates

The initial upstream ceilings are strict inequalities:

| Boundary | Required ceiling |
| --- | --- |
| BF16 DiT block | relative max and relative L2 below `1e-2` |
| Qwen layer-50 prompt output | relative max below `0.1`, relative L2 below `0.05` |
| 256x256x22 semantic latent | relative max below `0.15`, relative L2 below `0.10` |
| 512x512x22 two-step semantic latent | relative max and relative L2 below `0.08` |
| VisualVAE | relative max and relative L2 below `0.05` |
| AudioVAE | max absolute below `1e-3`, relative L2 below `0.05` |
| AudioVAE selected STFT magnitudes | relative max below `0.1`, relative L2 below `0.08` |

All numerical comparisons also require equal shapes and zero non-finite values.
An implementation issue may freeze a tighter measured boundary, but not a
looser one.

The semantic-latent gates first verify provenance rather than relying on
filenames or directory placement. Teacher and native manifests must agree on
the immutable model/reference revisions, seed, noise mode, internal geometry,
evaluation count, active block count, reuse interval, and SHA-256 of the exact
layer-50 BF16 conditioning payload. Only then are final video/audio latent
metrics evaluated. These gates are used when work crosses block, sampler, or
schedule boundaries. Attention-local iterations use the frozen single-block
teacher plus one 1,872-row block profile; they do not run 50 blocks or a
denoising trajectory. There is no routine 50-step development gate.

Complete 50-step `exact` generation belongs to end-user operation or rare,
explicit release validation. The checkerboard MP4 captured from the older
scalar binary is invalid quality evidence: its container is valid, but its
frames are a 32x32 grid of 16-pixel VAE cells produced from a noise-like
latent. It must not be used as a baseline or promotion artifact.

## Audiovisual Report

End-to-end reports include:

- video/audio latent relative max, relative L2, and non-finite counts;
- first/middle/last PSNR, pinned windowed SSIM, and pinned LPIPS;
- full-clip SSIM/LPIPS aggregates and adjacent-frame-delta error;
- waveform maximum/relative-L2 error and deterministic spectrogram error;
- independent left/right channel results, 32 kHz duration, 24 fps duration,
  and A/V sync;
- subject, count, anatomy, motion, composition, color, and prompt-adherence
  review.

`tools/strix/h3_quality.py` includes dependency-light numerical, global-SSIM,
Gaussian-window SSIM, temporal, and spectrogram diagnostics used by CI
corruption tests. `tools/strix/h3_preset_quality.py` additionally evaluates
AlexNet LPIPS from the pinned Nix toolchain. Every promoted report records the
NumPy/LPIPS/Torch/Torchvision versions and a canonical SHA-256 of the loaded
LPIPS module state. The Nix shell also pins the official Torchvision AlexNet
weights and LPIPS v0.1 calibration weights by complete SHA-256; the reporter
verifies both files before model construction, so evaluation cannot silently
download or substitute weights. Promotion reports require both sibling `parameters.json`
documents, validate their immutable model/reference revisions and frozen
preset values, bind those values to the delivered media geometry, and check
both decoded and FFprobe-reported stream timing. The
`--allow-missing-parameters` and `--skip-lpips` switches are diagnostic only.
The delivery spectrogram uses the same 1,024-point, 256-hop, centered
reflect-padded periodic-Hann STFT as the pinned AudioVAE oracle, including the
last samples of the waveform.

## Ownership by Implementation Boundary

Issues that add H3 runtime boundaries also add their retained artifact:

- loader/runtime: schema and lifecycle failure tests;
- tokenizer/Qwen: exact IDs and layer-50 artifact;
- host sampler: schedules/layout/RNG/Euler fixtures;
- DiT kernels: complete block artifact;
- full denoiser: velocity plus 256x256 four-step and 512x512 two-step latent
  artifacts, invoked only for changes crossing the denoiser boundary;
- VisualVAE: selected and full frame artifacts;
- AudioVAE: waveform, spectrogram, channel, duration, and sync artifacts;
- presets/optimization/quantization: component and short production-shape
  reports against BF16.

Fast and quantized routes are never compared only with each other.
