# MiniMax H3 Integration Boundary

Status: active engineering policy, 2026-08-21

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
./build-h3-169/minimax_h3_dit_hip_test --analytic

STRIX_H3_MODEL_ROOT=/var/llms/huggingface/MiniMax-H3 \
STRIX_H3_DIT_GOLDEN=/var/llms/huggingface/strix-h3-oracles/dit-block0-528-v1 \
  ./build-h3-169/minimax_h3_dit_hip_test --real
```

Sparse attention, token reduction, cross-block fusion, reuse, layer thinning,
and quantized DiT weights are not part of this baseline.

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
