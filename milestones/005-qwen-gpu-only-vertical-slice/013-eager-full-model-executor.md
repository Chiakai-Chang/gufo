---
id: M005-C013
title: "Compose the eager Qwen3.5-0.8B full-model prefill executor"
milestone: M005
status: planned
dependencies: [M005-C003, M005-C010, M005-C011, M005-C012, M005-C019]
---

# M005-C013: Compose the eager Qwen3.5-0.8B full-model prefill executor

## Dependencies

- [M005-C003](003-transactional-text-artifact-loader.md)
- [M005-C010](010-contiguous-kv-attention.md)
- [M005-C011](011-qwen-mlp-block.md)
- [M005-C012](012-lm-head-greedy-argmax.md)
- [M005-C019](019-qwen-deltanet-state-transactions.md)

## Required Context

- The vertical slice needs one native execution graph respecting the hybrid 3+1 block schedule, exact state boundaries, and explicit request transitions.

## Goal

Compose embedding, the exact 24-block hybrid schedule, final normalization, and LM head into one eager GPU-only prompt-prefill execution that produces validated final-position logits and provisional model state.

## Non-Goals

- HIP graphs
- Batching
- One-token decode or autoregressive generation; M005-C021 owns it.
- Greedy selection; M005-C020 owns it.
- Cancellation and final state commit.
- Sampling, chat, HTTP, HIP graphs, or NPU fallback.

## Expected Paths

- read: `docs/ROADMAP.md`
- read: `docs/GPU_BACKEND.md`
- read: `docs/TOKENIZATION.md`
- write: `models/qwen35_08b/executor.*`
- write: `models/qwen35_08b/runtime/prefill_executor.*`
- write: `tests/models/qwen35/full_model_test.*`

## Definition of Done

- [ ] The executor follows exactly 18 DeltaNet and 6 full-attention blocks from the descriptor.
- [ ] The executor follows exactly 18 DeltaNet and 6 full-attention blocks from the descriptor for every prompt token/chunk.
- [ ] Final-position full-vocabulary logits match the pinned oracle and all intermediate state remains provisional.
- [ ] Every dispatch is eager and uses preallocated storage; graph code is absent or disabled.
- [ ] Empty, minimum, representative, and configured maximum prompt lengths have explicit checked behavior.
- [ ] No token selection, decode iteration, or canonical state commit occurs in this card.

## Development Loop

```sh
mkdir -p artifacts/m005
nix develop -c cmake --preset hip-test
nix develop -c cmake --build --preset hip-test
git add -A
nix build
nix develop -c ctest --preset hip-test --output-on-failure -R 'qwen35_full_model_prefill'
```

## Hardware Validation

```sh
STRIX_ARTIFACT_DIR=artifacts/m005 STRIX_REQUIRE_GFX1151=1 nix develop -c ctest --preset hip-test --output-on-failure -R 'qwen35_full_model_prefill_hardware'
```

## Output Artifacts

- Full-model prefill execution trace.
- Per-layer dispatch/order and provisional-state record.
- Final-position full-logit comparison.

## Stop Conditions

- Stop at the first layer/state boundary mismatch; do not proceed to terminal UX.
- Stop if any NPU route, implicit fallback, or graph replay is selected.
- Stop before adding decode iteration, token commit, termination policy, or CLI behavior.

## ROADMAP Traceability

- tasks: M4.2 all model-private operations
- tasks: M4.3 contiguous KV integration
- tasks: M4.10 eager first
- exitCriteria: Artifact produces text prerequisite
- exitCriteria: Layer outputs and logits pass thresholds
- exitCriteria: No timed allocation
- docs: docs/ROADMAP.md#Milestone-4:-Qwen-GPU-Only-Vertical-Slice

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
