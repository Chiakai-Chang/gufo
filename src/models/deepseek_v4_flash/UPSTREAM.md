# DeepSeek V4 Flash Upstream

The model-private DeepSeek V4 Flash engine is adapted from DS4 commit
`84cc882352757baf628a1776badf7cc54d584e28`.

Only the model graph, GGUF/tokenizer support, request-session state, and ROCm
kernel dependency closure are retained. DS4 command-line, HTTP server, agent,
evaluation, and disk-cache frontends are intentionally excluded so Strix owns
those product surfaces.

The imported backend was converted to native ROCm/HIP source names and APIs.
It does not call Qwen kernels or share Qwen tensor layouts, dispatch policies,
or mutable state.

## Integration Boundary

`engine.hpp` and `engine.cpp` are the stable Strix-owned C++20 facade. Product
code uses that facade for model loading, tokenization, request sessions,
snapshots, logits, and cancellation. It does not include vendor headers.

The imported implementation is part of the Strix model package rather than a
nested external project:

- `runtime` owns the GGUF loader, tokenizer, graph, and request state.
- `kernels/rocm` owns every DeepSeek numerical kernel and ROCm dispatch rule.
- `engine.hpp` and `engine.cpp` are the stable C++ product facade.

The model package intentionally duplicates numerical code instead of calling
Qwen kernels. A DeepSeek kernel change must not alter another model's output or
performance.

The imported runtime and kernels retain their upstream formatting where that
keeps license review and numerical comparison auditable. New product behavior
belongs in the facade unless it is intrinsically part of the DeepSeek graph or
kernel implementation.

The standalone distributed, tensor-parallel, SSD planning/streaming,
multi-GPU placement, and DS4 frontend implementations are not built or
shipped. The retained upstream graph and ROCm translation units still contain
shared conditional ABI branches because the resident single-device route uses
the same internal types and kernels. The Strix facade cannot enable those
modes, and fail-closed compatibility hooks reject them if one is reached.

## Update Policy

When updating the imported engine:

1. Record the new immutable upstream commit.
2. Import only code exercised by the single-device ROCm route for the
   supported artifact.
3. Preserve the ROCm-only source and naming contract.
4. Reapply model-private changes inside this directory, never in Qwen code.
5. Run the DeepSeek model tests, the fast hardware suite, and the canonical
   Nix gate before recording new performance.
