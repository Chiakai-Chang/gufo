---
id: M005-C006
title: "Implement Qwen embedding, RMSNorm, residual, and RoPE HIP operations"
milestone: M005
status: planned
dependencies: [M003-C008, M005-C005]
---

# M005-C006: Implement Qwen embedding, RMSNorm, residual, and RoPE HIP operations

## Dependencies

- [M003-C008](../003-test-oracles-model-contracts/008-high-precision-cpu-operator-oracles.md)
- [M005-C005](005-request-state-and-gpu-arenas.md)

## Required Context

- These model-private operations establish raw-pointer HIP interfaces, model-local numerical ownership, tail handling, and reusable activation buffers.

## Goal

Implement eager gfx1151 kernels for embedding lookup, RMS normalization, residual add, and rotary position application with CPU-oracle agreement at declared boundaries.

## Non-Goals

- Attention
- GEMV/GEMM
- Kernel fusion beyond measured correctness-neutral tails
- HIP graphs
- 27B execution

## Expected Paths

- read: `docs/GPU_BACKEND.md`
- read: `docs/TESTING.md`
- write: `models/qwen35_08b/gpu/gfx1151/elementwise.hip`
- write: `models/qwen35_08b/cpu/oracles/`
- write: `tests/kernels/qwen35_elementwise_test.*`

## Definition of Done

- [ ] Minimum, typical 0.8B, tail, invalid-index, and boundary-position fixtures pass CPU-oracle thresholds.
- [ ] Kernels use explicit streams, raw pointers, POD dimensions, and preallocated scratch.
- [ ] Generated code is gfx1151-specific and model-private.
- [ ] The 27B descriptor validates launch dimensions only; no 27B dispatch occurs.

## Development Loop

```sh
mkdir -p artifacts/m005
nix develop -c cmake --preset hip-test
nix develop -c cmake --build --preset hip-test
git add -A
nix build
nix develop -c ctest --preset hip-test --output-on-failure -R 'qwen35_cpu_elementwise'
```

## Hardware Validation

```sh
STRIX_ARTIFACT_DIR=artifacts/m005 STRIX_REQUIRE_GFX1151=1 nix develop -c ctest --preset hip-test --output-on-failure -R 'm005_c006_hardware'
```

## Output Artifacts

- CPU-vs-HIP numerical report
- Critical kernel ISA/resource report

## Stop Conditions

- Stop on any non-finite value or unexplained tolerance increase.
- Stop if a kernel is placed in a shared numerical kernel registry.

## ROADMAP Traceability

- tasks: M4.2 embeddings, RMS normalization, RoPE, residual paths
- exitCriteria: Layer outputs pass oracle thresholds
- docs: docs/GPU_BACKEND.md#Model-Isolation
- docs: docs/TESTING.md#T2:-Device-kernel-tests

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
