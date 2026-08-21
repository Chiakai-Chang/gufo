# DeepSeek V4 Flash Q2-imatrix on Strix Halo

Status: 2026-08-21. This page is the current functional and performance
snapshot, not an optimization history.

## Model

| Field | Value |
| --- | --- |
| Repository | `antirez/deepseek-v4-gguf` |
| Snapshot | `1cd7b564460821938add0475a60b942c409295e0` |
| Artifact | `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` |
| Size | 80.76 GiB |
| Engine source | `antirez/ds4` at `84cc882352757baf628a1776badf7cc54d584e28` |
| Backend | Model-private ROCm/HIP implementation for `gfx1151` |

The DeepSeek graph, quantized layouts, session state, dispatch, and numerical
kernels live under `src/models/deepseek_v4_flash`. They do not call Qwen
kernels or share mutable Qwen state.

## Run

Build through the repository flake and define the artifact path:

```sh
git add .
nix build

MODEL=/path/to/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
```

Run a terminal prompt:

```sh
./result/bin/strix-server prompt \
  --model "$MODEL" \
  --raw-prompt \
  --prompt 'The capital of France is' \
  --max-tokens 16 \
  --temperature 0
```

Run the sparse long-context benchmark. Each prompt row adds a 2K suffix to the
prepared depth, and each generation row produces 128 autoregressive tokens.
Frontiers are extended incrementally and restored from in-memory snapshots.

```sh
./result/bin/strix-bench \
  --model "$MODEL" \
  --n-prompt 2048 \
  --n-gen 128 \
  --n-depth 2048,8192,16384,32768,65536 \
  --repetitions 1 \
  --verbose
```

The same artifact can be served through the existing OpenAI-compatible
completion and chat endpoints:

```sh
./result/bin/strix-server serve \
  --model "$MODEL" \
  --host 127.0.0.1 \
  --port 8080
```

## Current Results

The Strix rows use the release package, one repetition, a 2K prompt suffix, and
128 generated tokens. The DS4 reference is the supplied same-machine result
for the same artifact. Prompt rows compare the final context after adding the
2K suffix; generation rows compare the prepared depth.

| Prepared depth | Strix `pp2048` | DS4 `pp2048` | Delta | Strix `tg128` | DS4 `tg128` | Delta | Snapshot bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 193.66 | 205.49 | -5.8% | 15.52 | 14.76 | +5.1% | 52,184,460 |
| 8K | 188.28 | 197.13 | -4.5% | 14.61 | 13.87 | +5.3% | 136,750,476 |
| 16K | 183.10 | 190.09 | -3.7% | 14.32 | 13.63 | +5.1% | 249,505,164 |
| 32K | 167.05 | 171.83 | -2.8% | 13.55 | 12.93 | +4.8% | 475,014,540 |
| 64K | 145.66 | not supplied | - | 12.42 | 11.91 | +4.3% | 926,033,292 |

The model loads 80.76 GiB of tensor spans in about 21 seconds. The 64K run
plans 82.07 GiB total, including model, KV state, and working buffers.
The 2K row was rerun after the native C++ runtime refactor; the deeper rows
retain the prior measurements from the same numerical kernel family.

## Quality and Integration

- The pinned upstream DS4 CLI and packaged Strix CLI produce the exact same
  four-token greedy continuation, ` Paris. It is`, for the same raw prompt and
  artifact.
- Full logits are finite after real GGUF prefill.
- Snapshot restore reproduces the exact position, greedy token, and logit
  vector.
- Direct model sessions and the HTTP backend produce identical raw and chat
  token sequences.
- Cancellation returns the request session cleanly for exact subsequent reuse.
- Terminal prompt, benchmark, raw completion, and chat completion use the same
  model-owned engine and tokenizer.

## Successful

- Adapted the required DS4 graph and ROCm kernels into repository-owned C++20
  `runtime` and `kernels/rocm` packages.
- Removed standalone distributed, tensor-parallel, SSD weight streaming,
  multi-GPU placement, CPU reference, MTP, steering, and embedded hotlist
  implementations from the production closure.
- Converted the private fork to native ROCm/HIP naming and APIs.
- Kept DeepSeek code, kernels, state, and dispatch isolated from Qwen.
- Reused the existing Strix CLI, benchmark, and OpenAI-compatible server.
- Matched the DS4 state payload sizes and stayed close to or ahead of the
  supplied throughput curve through 64K.

## Failed

- Automatic CMake discovery was not reliable for the header-only ROCm
  dependencies in a clean Nix sandbox. Explicit flake-provided include roots
  are retained.
- The upstream DS4 frontend build is not imported. Strix owns CLI, HTTP,
  benchmarking, cancellation, and session pooling.

## To Do

- Add model-owned thinking/reasoning mode and effort controls; the current chat
  template intentionally uses the no-thinking path.
- Run the compact `strix-eval` qualification suite when #153 is implemented.
- Profile and optimize the model-owned gfx1151 kernels under #155.
- Add DSpark speculative decoding under #156.
- Add model-owned offline calibration/imatrix tooling only when a new
  quantization recipe requires it.
- Add multi-session scheduling and restart-safe SSD state reuse through the
  serving milestones.
