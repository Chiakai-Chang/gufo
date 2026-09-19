# Third-Party Notices and Inventory

This document records the complete inventory of third-party software, libraries, drivers,
and system components used, linked, or required by **gufo**.

The engine license is in [LICENSE](LICENSE); model-specific acquisition and
usage records are linked from [the model guide](docs/models/README.md).

---

## Inventory Summary

| Component Name | Relationship | License (SPDX) | Pinned Revision / Version | Upstream Source / Location |
| --- | --- | --- | --- | --- |
| **ROCm / HIP** | Linked / Toolchain | `MIT OR Apache-2.0 WITH LLVM-exception` | Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [ROCm/clr](https://github.com/ROCm/clr) |
| **hipBLAS** | Linked | `MIT` | Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [ROCm/hipBLAS](https://github.com/ROCm/hipBLAS) |
| **hipBLASLt** | Linked | `MIT` | Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [ROCm/hipBLASLt](https://github.com/ROCm/hipBLASLt) |
| **rocBLAS** | Linked | `MIT` | Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [ROCm/rocBLAS](https://github.com/ROCm/rocBLAS) |
| **Composable Kernel** | Header / compiled kernels | `MIT` | Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [ROCm/composable_kernel](https://github.com/ROCm/composable_kernel) |
| **ROCprofiler SDK / ROCTx** | Benchmark marker library / profiling tool | `MIT` | Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [ROCm/rocprofiler-sdk](https://github.com/ROCm/rocprofiler-sdk) |
| **DS4** | Vendored model engine and ROCm kernels | `MIT` | `84cc882352757baf628a1776badf7cc54d584e28` | [antirez/ds4](https://github.com/antirez/ds4) |
| **h3.c** | Pinned implementation reference; selected code may be adapted model-privately | `MIT` | `8974cc055ea9c02fcd14cc27dfda3e1027c05153` | [antirez/h3.c](https://github.com/antirez/h3.c) |
| **ccv TensorOps matmul ancestry** | Algorithm/source ancestry identified by h3.c | `BSD-3-Clause` | Notice pinned through h3.c commit `8974cc055ea9c02fcd14cc27dfda3e1027c05153` | [libccv/ccv](https://github.com/liuliu/ccv) |
| **llama.cpp** | Source-derived algorithm | `MIT` | `e9fa0781f1c25fc4fe8c86be1edc6970661ad6f0` | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| **ICU** | Linked | `Unicode-3.0` | Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [unicode-org/icu](https://github.com/unicode-org/icu) |
| **curl / libcurl** | Linked HTTP client | `curl` | `8.21.0`, Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [curl/curl](https://github.com/curl/curl) |
| **FFmpeg** | Spawned runtime executable | `LGPL-2.1-or-later AND GPL-2.0-or-later` (enabled components may also be `LGPL-3.0-or-later` / `GPL-3.0-or-later`) | `8.1.2`, Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [FFmpeg/FFmpeg](https://github.com/FFmpeg/FFmpeg) |
| **PyTorch ROCm** | Evaluation/offline-teacher tool only; not shipped | `BSD-3-Clause` | `2.12.0`, Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [pytorch/pytorch](https://github.com/pytorch/pytorch) |
| **Torchvision** | Evaluation/LPIPS tool only; not shipped | `BSD-3-Clause` | `0.27.0`, Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [pytorch/vision](https://github.com/pytorch/vision) |
| **LPIPS** | Perceptual-quality evaluation tool only; not shipped | `BSD-2-Clause` | `0.1.4`, Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity) |
| **Torchvision AlexNet weights** | Evaluation model data only; not shipped in the production package | `NOASSERTION` | SHA-256 `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02` | [PyTorch model distribution](https://download.pytorch.org/models/alexnet-owt-7be5be79.pth) |
| **MiniMax H3 FL2VA checkpoint** | External operator-supplied model; not distributed | `LicenseRef-MiniMax-H3-Community-2026-08-02` or operator-specific authorization | `42ed227ee7df40d41602854ae760620d6eb651fe` | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) |
| **Qwen3-VL-32B encoder weights used by H3** | External operator-supplied model component; not distributed | `Apache-2.0` | Included by the pinned H3 FL2VA checkpoint | [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) |

---

## Distributed Userspace Dependencies

### ROCm / HIP (AMD ROCm Compute Language Runtime)

- **Component Name**: ROCm / HIP (AMD ROCm Compute Language Runtime)
- **Upstream URL**: https://github.com/ROCm/clr
- **Pinned Revision**: Nixpkgs `nixos-unstable` lock revision `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` (`rocmPackages.clr` / `rocmPackages.llvm.clang`)
- **Component Used**: HIP runtime, headers, host/device headers, and Clang compiler for `gfx1151`
- **SPDX License Identifier**: `MIT OR Apache-2.0 WITH LLVM-exception`
- **Copyright / Notice Source**: Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
- **Relationship**: Linked / Toolchain (Build toolchain and runtime library dependency)
- **Corresponding-Source Location**: https://github.com/ROCm/clr (via Nix derivation `rocmPackages.clr`)

### hipBLAS

- **Component Name**: hipBLAS (AMD ROCm BLAS Marshalling Library)
- **Upstream URL**: https://github.com/ROCm/hipBLAS
- **Pinned Revision**: Nixpkgs `nixos-unstable` lock revision `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` (`rocmPackages.hipblas`)
- **Component Used**: Dynamic BLAS abstraction library and interface headers
- **SPDX License Identifier**: `MIT`
- **Copyright / Notice Source**: Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
- **Relationship**: Linked (Dynamic library dependency)
- **Corresponding-Source Location**: https://github.com/ROCm/hipBLAS (via Nix derivation `rocmPackages.hipblas`)

### hipBLASLt

- **Component Name**: hipBLASLt (AMD ROCm Tunable BLAS Library)
- **Upstream URL**: https://github.com/ROCm/hipBLASLt
- **Pinned Revision**: Nixpkgs `nixos-unstable` lock revision `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` (`rocmPackages.hipblaslt`)
- **Component Used**: Tuned BF16 matrix multiplication kernels and algorithm-selection interface for prompt processing
- **SPDX License Identifier**: `MIT`
- **Copyright / Notice Source**: Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
- **Relationship**: Linked (Dynamic library dependency)
- **Corresponding-Source Location**: https://github.com/ROCm/hipBLASLt (via Nix derivation `rocmPackages.hipblaslt`)

### rocBLAS

- **Component Name**: rocBLAS (AMD ROCm Basic Linear Algebra Subprograms)
- **Upstream URL**: https://github.com/ROCm/rocBLAS
- **Pinned Revision**: Nixpkgs `nixos-unstable` lock revision `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` (`rocmPackages.rocblas`)
- **Component Used**: GPU BLAS execution kernels optimized for `gfx1151`
- **SPDX License Identifier**: `MIT`
- **Copyright / Notice Source**: Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
- **Relationship**: Linked (Dynamic library dependency)
- **Corresponding-Source Location**: https://github.com/ROCm/rocBLAS (via Nix derivation `rocmPackages.rocblas`)

### Composable Kernel

- **Component Name**: Composable Kernel (ROCm GPU kernel library)
- **Upstream URL**: https://github.com/ROCm/composable_kernel
- **Pinned Revision**: Nixpkgs `nixos-unstable` lock revision `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` (`rocmPackages.composable_kernel`)
- **Component Used**: Header-instantiated causal grouped-query attention kernel compiled into the `gfx1151` HIP backend
- **SPDX License Identifier**: `MIT`
- **Copyright / Notice Source**: Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
- **Relationship**: Header / compiled kernels
- **Corresponding-Source Location**: https://github.com/ROCm/composable_kernel (via Nix derivation `rocmPackages.composable_kernel`)

### ROCprofiler SDK / ROCTx

- **Component Name**: ROCprofiler SDK / ROCTx
- **Upstream URL**: https://github.com/ROCm/rocprofiler-sdk
- **Pinned Revision**: Nixpkgs `nixos-unstable` lock revision `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` (`rocmPackages.rocprofiler-sdk`)
- **Component Used**: The `rocprofv3` diagnostic profiler and the lightweight ROCTx marker library linked only by `gufo-kernel-bench`
- **SPDX License Identifier**: `MIT`
- **Copyright / Notice Source**: Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
- **Relationship**: Benchmark marker library / profiling tool; the inference server does not link the profiler SDK
- **Corresponding-Source Location**: https://github.com/ROCm/rocprofiler-sdk (via Nix derivation `rocmPackages.rocprofiler-sdk`)

### ICU

- **Component Name**: ICU (International Components for Unicode)
- **Upstream URL**: https://github.com/unicode-org/icu
- **Pinned Revision**: Nixpkgs `nixos-unstable` lock revision
  `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` (`icu`)
- **Component Used**: Unicode NFC normalization and Unicode general-category /
  whitespace classification for the model-private MiniMax H3 tokenizer
- **SPDX License Identifier**: `Unicode-3.0`
- **Relationship**: Linked dynamic runtime dependency
- **Corresponding-Source Location**: https://github.com/unicode-org/icu
  (via Nix derivation `icu`)

### FFmpeg

- **Component Name**: FFmpeg
- **Upstream URL**: https://github.com/FFmpeg/FFmpeg
- **Pinned Version**: `8.1.2` from Nixpkgs lock revision
  `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a`
- **Component Used**: Headless `ffmpeg` and `ffprobe` executables for
  concurrent RGB24/F32 PCM input, H.264/AAC MP4 encoding, and output metadata
  validation
- **SPDX License Identifiers**: `LGPL-2.1-or-later` and
  `GPL-2.0-or-later`; the pinned Nix derivation also declares
  `LGPL-3.0-or-later` and `GPL-3.0-or-later` for enabled optional components
- **Relationship**: Spawned as a separate runtime process through pipes.
  Gufo does not link FFmpeg libraries or copy FFmpeg source into the engine.
- **Corresponding-Source Location**: https://github.com/FFmpeg/FFmpeg
  (via Nix derivation `ffmpeg-headless`)

The Nix closure retains FFmpeg's complete license and corresponding-source
metadata. H.264/AAC availability and any patent obligations are deployment
considerations separate from the MiniMax model license and the engine's MIT
license.

### llama.cpp

- **Component Name**: llama.cpp
- **Upstream URL**: https://github.com/ggml-org/llama.cpp
- **Pinned Revision**: Commit `e9fa0781f1c25fc4fe8c86be1edc6970661ad6f0`
- **Component Used**: Causal tiled-attention scheduling and online-softmax algorithm adapted into the native Qwen3.8 `gfx1151` HIP kernel
- **SPDX License Identifier**: `MIT`
- **Copyright / Notice Source**: Copyright (c) 2023-2026 The ggml authors
- **Relationship**: Source-derived algorithm; no llama.cpp runtime code or library is linked
- **Corresponding-Source Location**: https://github.com/ggml-org/llama.cpp/tree/e9fa0781f1c25fc4fe8c86be1edc6970661ad6f0/ggml/src/ggml-cuda

### DS4

- **Component Name**: DS4
- **Upstream URL**: https://github.com/antirez/ds4
- **Pinned Revision**: Commit `84cc882352757baf628a1776badf7cc54d584e28`
- **Component Used**: DeepSeek V4 Flash GGUF loader, tokenizer, request-session
  graph, and ROCm numerical kernels
- **SPDX License Identifier**: `MIT`
- **Copyright / Notice Source**: Copyright (c) 2026 Salvatore Sanfilippo
  and DS4 contributors; the upstream license is retained under
  `src/models/deepseek_v4_flash/vendor/antirez/LICENSE`.
- **Relationship**: Vendored and adapted into a model-private ROCm backend.
  Gufo does not import the upstream command-line interface, HTTP server,
  agent, evaluator, or disk-cache frontend.
- **Corresponding-Source Location**:
  https://github.com/antirez/ds4/tree/84cc882352757baf628a1776badf7cc54d584e28
- **Official continuation fixtures**: The matching 0731 hosted-model
  continuations and five smoke prompts are imported separately from revision
  `6289c516273979173abbc062209a81dd3706b804`. The fixture
  `tests/models/deepseek_v4_flash/fixtures/official-0731.json` retains source
  hashes and the upstream MIT notice (the ds4.c authors and ggml authors).

### h3.c

- **Component Name**: h3.c
- **Upstream URL**: https://github.com/antirez/h3.c
- **Pinned Revision**: `8974cc055ea9c02fcd14cc27dfda3e1027c05153`
- **Component Used**: Native MiniMax H3 architecture, tensor naming, packed
  multimodal layout, scheduler, sampler, VAE, tokenizer, fixtures, and
  performance reference for the model-private ROCm/HIP port
- **SPDX License Identifier**: `MIT`
- **Copyright / Notice Source**: Copyright (c) 2026 Salvatore Sanfilippo
- **Relationship**: Pinned implementation reference. Selected compatible code
  may be copied or adapted into `src/models/minimax_h3/` with provenance and
  modifications recorded; Metal and Objective-C execution are not imported.
- **Corresponding-Source Location**:
  https://github.com/antirez/h3.c/tree/8974cc055ea9c02fcd14cc27dfda3e1027c05153

The upstream `THIRD_PARTY_NOTICES.md` identifies the rectangular Morton
decoder and dynamic INT8/TensorOps scheduling design in `h3_shaders.metal` as
adapted from ccv's `NAMatMulKernel` and `NAInt8MatMulKernel`, licensed
BSD-3-Clause with copyright (c) 2010, Liu Liu. Any adapted expression or design
retains that notice. See
[the MiniMax H3 provenance record](docs/models/minimax-h3/EVALUATION.md).

### curl / libcurl

- **Component Name**: curl / libcurl
- **Upstream URL**: https://github.com/curl/curl
- **Pinned Revision**: Version `8.21.0` from Nixpkgs lock revision
  `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a`
- **Component Used**: HTTPS-capable client library used only by `gufo eval`
  for OpenAI-compatible model discovery and chat-completion requests
- **SPDX License Identifier**: `curl`
- **Relationship**: Linked dynamic runtime dependency
- **Corresponding-Source Location**: https://github.com/curl/curl

---

## External Model Artifacts (Non-Distributed)

Model checkpoints are not part of the gufo source or binary
distribution. Operators obtain them directly from their publisher and remain
responsible for the terms governing their location and use.

### MiniMax H3 FL2VA

- **Component Name**: MiniMax H3 Base FL2VA checkpoint
- **Upstream URL**: https://huggingface.co/MiniMaxAI/MiniMax-H3
- **Pinned Revision**: `42ed227ee7df40d41602854ae760620d6eb651fe`
- **Component Used**: Locally supplied BF16 text encoder, Omni Transformer,
  VisualVAE, AudioVAE, tokenizer, scheduler configuration, and metadata under
  `FL2VA/`
- **License**: MiniMax H3 Community License Agreement dated August 2, 2026, or
  separate operator-specific authorization where required
- **License File SHA-256 at Pinned Revision**:
  `59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44`
- **Copyright / Notice Source**: Copyright © 2026 MiniMax. All Rights Reserved.
- **Relationship**: External, access-controlled, operator-supplied runtime
  artifact. It is never committed, packaged, mirrored, automatically
  downloaded, or redistributed by gufo.
- **Operational Boundary**: Every operator must independently obtain access
  from MiniMax, accept or obtain the terms applicable to that operator, and
  configure a local checkpoint path. Possession of the gufo source
  does not grant model rights.
- **Serving Boundary**: The engine supplies numerical execution only. Anyone
  exposing H3 through an API is responsible for the publisher's user terms,
  acceptable-use, safeguards, disclosures, reporting, attribution, and
  territorial requirements.

The project records an operator attestation that the dedicated development
machine is authorized for this work. The repository does not contain private
license correspondence or credentials and does not independently make a legal
determination about a downstream operator.

### Qwen3-VL-32B Encoder Component

- **Component Name**: Qwen3-VL-32B encoder weights used by MiniMax H3
- **Upstream URL**: https://github.com/QwenLM/Qwen3-VL
- **Pinned Revision**: Supplied as part of the pinned MiniMax H3 FL2VA package
- **Component Used**: H3 prompt encoder through layer 50
- **SPDX License Identifier**: `Apache-2.0`
- **Relationship**: External model component within the operator-supplied H3
  checkpoint; not distributed by gufo

See [the MiniMax H3 guide](docs/models/minimax-h3/README.md) for the acquisition, release, and
runtime boundary.

---

## Evaluation-Only Quality Toolchain (Non-Shipped)

The following packages are present only in the pinned Nix development shell.
They are not linked into, copied into, or distributed with the production
`gufo` package.

### PyTorch ROCm

- **Component Name**: PyTorch with ROCm support
- **Upstream URL**: https://github.com/pytorch/pytorch
- **Pinned Version**: `2.12.0` from Nixpkgs lock revision
  `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a`
- **Component Used**: Independent MiniMax H3 teacher execution and tensor
  inspection on the supported ROCm host
- **SPDX License Identifier**: `BSD-3-Clause`
- **Relationship**: Evaluation and offline tooling only; absent from the
  production package closure

### Torchvision

- **Component Name**: Torchvision
- **Upstream URL**: https://github.com/pytorch/vision
- **Pinned Version**: `0.27.0` from Nixpkgs lock revision
  `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a`
- **Component Used**: AlexNet feature network used by the LPIPS evaluator
- **SPDX License Identifier**: `BSD-3-Clause`
- **Relationship**: Evaluation-only direct dependency of the pinned Python
  toolchain; absent from the production package closure

### LPIPS

- **Component Name**: Learned Perceptual Image Patch Similarity (LPIPS)
- **Upstream URL**: https://github.com/richzhang/PerceptualSimilarity
- **Pinned Version**: `0.1.4` from Nixpkgs lock revision
  `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a`
- **Component Used**: AlexNet-based perceptual comparison of delivered
  MiniMax H3 video frames
- **SPDX License Identifier**: `BSD-2-Clause`
- **Relationship**: Evaluation-only tool; absent from the production package
  closure
- **Model Boundary**: Feature-network parameters are not committed or
  redistributed by gufo. Each promoted quality report records a
  canonical SHA-256 of the loaded module state and the exact
  LPIPS/PyTorch/Torchvision versions.

### Torchvision AlexNet evaluation weights

- **Component Name**: Torchvision AlexNet ImageNet weights
- **Upstream URL**:
  https://download.pytorch.org/models/alexnet-owt-7be5be79.pth
- **Pinned Digest**:
  `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`
- **Component Used**: Frozen AlexNet feature trunk for LPIPS evaluation
- **SPDX License Identifier**: `NOASSERTION`
- **Relationship**: Nix-fetched evaluation model data only; absent from the
  production package closure and not committed to the repository

The evaluator verifies this digest and the bundled LPIPS v0.1 AlexNet
calibration digest
`df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0`
before constructing the metric. It fails closed when either file is absent or
changed and therefore does not rely on an ambient Torch hub cache or a runtime
download.
