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

`vendor/antirez` is a model-private upstream fork. Its model graph and
numerical kernels are deliberately isolated from the repository's shared core
and from every other model. The complete target definition lives in this
directory so model-specific sources, flags, and dependencies do not leak into
the root build.

The vendor files are not reformatted into the Strix style. Keeping their shape
close to the pinned source makes license review, upstream comparison, and
future updates auditable. New Strix behavior belongs in the facade unless it is
intrinsically part of the DeepSeek graph or kernel implementation.

## Update Policy

When updating the imported engine:

1. Record the new immutable upstream commit.
2. Import only the graph, tokenizer/GGUF, session-state, and ROCm dependency
   closure required by the supported artifact.
3. Preserve the ROCm-only source and naming contract.
4. Reapply model-private changes inside this directory, never in Qwen code.
5. Run the DeepSeek model tests, the fast hardware suite, and the canonical
   Nix gate before recording new performance.
