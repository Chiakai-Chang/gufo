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
