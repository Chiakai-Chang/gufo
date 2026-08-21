# MiniMax H3 Quality Oracles

Status: frozen foundation contract, 2026-08-21

## Rule

MiniMax H3 work is promoted in this order:

1. an independent analytic or pinned teacher oracle;
2. correctness and quality gates;
3. profiling;
4. optimization or quantization.

A successful kernel launch, finite latent, recognizable frame, or playable MP4
is not a quality result. The exact BF16 route is the reference for every later
fast or quantized route.

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

## Numerical Gates

The initial upstream ceilings are strict inequalities:

| Boundary | Required ceiling |
| --- | --- |
| BF16 DiT block | relative max and relative L2 below `1e-2` |
| Qwen layer-50 prompt output | relative max below `0.1`, relative L2 below `0.05` |
| 256x256x22 semantic latent | relative max below `0.15`, relative L2 below `0.10` |
| VisualVAE | relative max and relative L2 below `0.05` |
| AudioVAE | max absolute below `1e-3`, relative L2 below `0.05` |

All numerical comparisons also require equal shapes and zero non-finite values.
An implementation issue may freeze a tighter measured boundary, but not a
looser one.

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
temporal, and spectrogram diagnostics used by CI corruption tests. Promoted
reports must additionally name and pin the windowed-SSIM and LPIPS
implementations that produced release metrics.

## Ownership by Implementation Boundary

Issues that add H3 runtime boundaries also add their retained artifact:

- loader/runtime: schema and lifecycle failure tests;
- tokenizer/Qwen: exact IDs and layer-50 artifact;
- host sampler: schedules/layout/RNG/Euler fixtures;
- DiT kernels: complete block artifact;
- full denoiser: velocity, rapid latent, and 50-step exact latent artifacts;
- VisualVAE: selected and full frame artifacts;
- AudioVAE: waveform, spectrogram, channel, duration, and sync artifacts;
- presets/optimization/quantization: paired reports against BF16 exact.

Fast and quantized routes are never compared only with each other.
