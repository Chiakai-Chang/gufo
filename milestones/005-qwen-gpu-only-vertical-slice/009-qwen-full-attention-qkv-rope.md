---
id: M005-C009
title: "Implement full-attention QKV preparation and output path"
milestone: M005
status: planned
dependencies: [M003-C008, M005-C005, M005-C006, M005-C007]
---

# M005-C009: Implement full-attention QKV preparation and output path

## Dependencies

- [M003-C008](../003-test-oracles-model-contracts/008-high-precision-cpu-operator-oracles.md)
- [M005-C005](005-request-state-and-gpu-arenas.md)
- [M005-C006](006-qwen-elementwise-hip-ops.md)
- [M005-C007](007-qwen-projection-hip-ops.md)

## Required Context

- Every fourth 0.8B block is full attention. This card owns Q/K/V projection orchestration, model-specific head mapping, RoPE application, and output projection boundaries, but not KV storage or attention reduction.

## Goal

Produce and validate Q, K, and V buffers and consume an oracle attention result through the output projection for all six 0.8B full-attention layers.

## Non-Goals

- KV cache allocation
- Attention softmax/reduction
- Paged attention
- Flash attention claims
- NPU route

## Expected Paths

- read: `docs/GPU_BACKEND.md#Attention`
- read: `docs/TESTING.md`
- write: `models/qwen35_08b/gpu/gfx1151/full_attention_prepare*.hip`
- write: `models/qwen35_08b/full_attention.*`
- write: `tests/models/qwen35/full_attention_prepare_test.*`

## Definition of Done

- [ ] Q/K/V and post-RoPE boundaries match CPU/reference artifacts for minimum and supported context positions.
- [ ] Head/GQA mapping and position arithmetic are checked before launch.
- [ ] All six full-attention layers bind correctly and DeltaNet layers cannot enter this dispatch.
- [ ] Output projection accepts only the declared attention layout.

## Development Loop

```sh
mkdir -p artifacts/m005
nix develop -c cmake --preset hip-test
nix develop -c cmake --build --preset hip-test
git add -A
nix build
nix develop -c ctest --preset hip-test --output-on-failure -R 'qwen35_full_attention_prepare'
```

## Hardware Validation

```sh
STRIX_ARTIFACT_DIR=artifacts/m005 STRIX_REQUIRE_GFX1151=1 nix develop -c ctest --preset hip-test --output-on-failure -R 'm005_c009_hardware'
```

## Output Artifacts

- Q/K/V, rotary, and output boundary comparison JSON

## Stop Conditions

- Stop on a head-count, GQA mapping, or RoPE convention mismatch.
- Stop before combining preparation and attention kernels if independent boundaries do not pass.

## ROADMAP Traceability

- tasks: M4.2 RoPE; Q/K/V and output projections
- exitCriteria: Layer outputs pass oracle thresholds
- docs: docs/GPU_BACKEND.md#Attention
- docs: docs/TESTING.md#T3:-Layer-and-state-boundary-tests

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
