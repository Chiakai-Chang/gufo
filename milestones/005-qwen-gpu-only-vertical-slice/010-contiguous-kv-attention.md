---
id: M005-C010
title: "Implement contiguous KV cache and eager prefill/decode attention"
milestone: M005
status: planned
dependencies: [M003-C008, M005-C005, M005-C009]
---

# M005-C010: Implement contiguous KV cache and eager prefill/decode attention

## Dependencies

- [M003-C008](../003-test-oracles-model-contracts/008-high-precision-cpu-operator-oracles.md)
- [M005-C005](005-request-state-and-gpu-arenas.md)
- [M005-C009](009-qwen-full-attention-qkv-rope.md)

## Required Context

- M4 explicitly starts with a contiguous request-local KV cache. Paged KV belongs to M7. GPU decode is the sole owner for MVP.

## Goal

Implement capacity-checked BF16 contiguous K/V storage and model-private causal prefill and one-token decode attention with position-safe append semantics.

## Non-Goals

- Paged KV
- Prefix sharing
- KV persistence
- KV quantization
- Continuous batching
- NPU reads

## Expected Paths

- read: `docs/KV_CACHE.md`
- read: `docs/GPU_BACKEND.md#Attention`
- write: `models/qwen35_08b/kv/contiguous.*`
- write: `models/qwen35_08b/gpu/gfx1151/attention*.hip`
- write: `models/qwen35_08b/cpu/oracles/attention.*`
- write: `tests/models/qwen35/attention_kv_test.*`

## Definition of Done

- [ ] Capacity and bytes-per-token are derived from the compiled descriptor with checked arithmetic.
- [ ] Prefill-then-decode agrees with token-serial recomputation across empty, short, boundary, and maximum fixture contexts.
- [ ] A transition reserves capacity before writing and cancellation/failure leaves valid length and KV bytes unchanged.
- [ ] Timed append and attention perform no general allocation.

## Development Loop

```sh
mkdir -p artifacts/m005
nix develop -c cmake --preset hip-test
nix develop -c cmake --build --preset hip-test
git add -A
nix build
nix develop -c ctest --preset hip-test --output-on-failure -R 'qwen35_(contiguous_kv|attention_cpu)'
```

## Hardware Validation

```sh
STRIX_ARTIFACT_DIR=artifacts/m005 STRIX_REQUIRE_GFX1151=1 nix develop -c ctest --preset hip-test --output-on-failure -R 'm005_c010_hardware'
```

## Output Artifacts

- KV byte/capacity model JSON
- Prefill-vs-serial attention and KV boundary artifact

## Stop Conditions

- Stop on any out-of-capacity write or position discontinuity.
- Stop if cancellation changes canonical KV valid length or payload.

## ROADMAP Traceability

- tasks: M4.2 decode and prefill attention
- tasks: M4.3 initial contiguous KV cache
- exitCriteria: Layer outputs pass thresholds
- exitCriteria: No general allocation in timed decode
- exitCriteria: Cancellation reclaims state
- docs: docs/KV_CACHE.md#Allocation
- docs: docs/TESTING.md#T3:-Layer-and-state-boundary-tests

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
