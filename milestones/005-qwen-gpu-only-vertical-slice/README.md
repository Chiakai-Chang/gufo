# M005: Qwen GPU-Only Vertical Slice

Roadmap source: [`docs/ROADMAP.md`, Milestone 4](../../docs/ROADMAP.md)

## Objective

Produce native, text-only, deterministic greedy terminal generation for the
pinned Qwen3.5-0.8B candidate artifact on gfx1151. Reuse checked descriptors for
Qwen3.8-27B without downloading or executing 27B. The route is eager and
GPU-only; NPU, HTTP, batching, paged KV, speculation, and HIP graphs are outside
this MVP.

## Cards

| Card | Purpose |
| --- | --- |
| [M005-C001](001-gpu-only-strix-server-target.md) | Wire the existing server target to the GPU runtime without command renames |
| [M005-C002](002-qwen-compiled-descriptors.md) | Bind validated descriptors to runtime dispatch |
| [M005-C003](003-transactional-text-artifact-loader.md) | Transactional text-only artifact loading |
| [M005-C004](004-native-qwen-tokenizer-template.md) | Bind validated tokenizer/template to runtime |
| [M005-C005](005-request-state-and-gpu-arenas.md) | Request state, arenas, allocation instrumentation |
| [M005-C006](006-qwen-elementwise-hip-ops.md) | Elementwise GPU operators |
| [M005-C007](007-qwen-projection-hip-ops.md) | Projection dispatch and remaining packed formats |
| [M005-C008](008-qwen-gated-deltanet-state.md) | DeltaNet numerical primitives |
| [M005-C019](019-qwen-deltanet-state-transactions.md) | DeltaNet state and transactional execution |
| [M005-C009](009-qwen-full-attention-qkv-rope.md) | QKV, RoPE, and full-attention primitives |
| [M005-C010](010-contiguous-kv-attention.md) | Contiguous KV prefill/decode state |
| [M005-C011](011-qwen-mlp-block.md) | MLP block |
| [M005-C012](012-lm-head-greedy-argmax.md) | Final norm and LM-head logits |
| [M005-C020](020-deterministic-greedy-argmax.md) | Device-side deterministic greedy argmax |
| [M005-C013](013-eager-full-model-executor.md) | Full-model eager prefill |
| [M005-C021](021-one-token-decode-generation-loop.md) | Transactional one-token decode and generation loop |
| [M005-C014](014-layer-logit-oracle-gates.md) | Layer and full-logit gates |
| [M005-C015](015-direct-generation-request.md) | Direct generation request lifecycle |
| [M005-C016](016-strix-server-prompt-cli.md) | Existing terminal prompt command |
| [M005-C017](017-exact-token-and-eval-drift.md) | Exact-token fixtures |
| [M005-C022](022-capability-drift-trace-regrade.md) | Capability drift trace and offline regrade |
| [M005-C018](018-m4-gfx1151-acceptance-baselines.md) | Final M4 acceptance and baselines |

## ROADMAP Task Coverage

| ROADMAP task | Cards |
| --- | --- |
| 1. Transactional model loading and checksums | M005-C003 |
| 2. Model-private Qwen GPU operations | M005-C006–C013, C019 |
| 3. Initial contiguous KV | M005-C005, C010 |
| 4. Deterministic greedy sampling | M005-C020, C021 |
| 5. Request-owned model/sampling state | M005-C005, C019–C021 |
| 6. Capability drift gate, trace, regrade | M005-C022 |
| 7. Direct terminal prompt | M005-C015, C016 |
| 8. Exact-token fixtures | M005-C017 |
| 9. Sampling and interactive chat after greedy | Explicitly deferred beyond this greedy MVP; no card claims completion |
| 10. Eager first; HIP graphs later | M005-C013, C021; graphs explicitly excluded |

## Exit-Criterion Coverage

| Exit criterion | Cards |
| --- | --- |
| Safetensors-derived artifact produces terminal text | M005-C003, C016, C017, C018 |
| Greedy token history deterministic | M005-C020, C021, C017 |
| Layer outputs and full logits pass thresholds | M005-C014, C018 |
| No general allocation during timed decode | M005-C005, C021, C018 |
| Direct cancellation reclaims provisional state | M005-C019, C021, C015, C018 |
| TTFT, inter-token latency, memory baselines retained | M005-C018 |

## Completion Rule

The greedy GPU MVP is complete when M005-C018 passes on the supported gfx1151
machine with all of its dependencies. Sampling, chat, HTTP, HIP graphs, paged
KV, NPU routing, and Qwen3.8-27B production validation remain explicitly
unimplemented rather than being silently credited to this milestone.
