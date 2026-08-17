---
id: M005-C002
title: "Bind the validated Qwen descriptors to GPU runtime dispatch"
milestone: M005
status: planned
dependencies: [M003-C001, M003-C003, M003-C004, M003-C005, M003-C013, M005-C001]
---

# M005-C002: Bind the validated Qwen descriptors to GPU runtime dispatch

## Dependencies

- [M003-C001](../003-test-oracles-model-contracts/001-checked-tensor-descriptor.md)
- [M003-C003](../003-test-oracles-model-contracts/003-compiled-model-kind-registry.md)
- [M003-C004](../003-test-oracles-model-contracts/004-qwen35-08b-descriptor.md)
- [M003-C005](../003-test-oracles-model-contracts/005-qwen38-27b-text-descriptor-reuse.md)
- [M003-C013](../003-test-oracles-model-contracts/013-milestone2-contract-gate.md)
- [M005-C001](001-gpu-only-strix-server-target.md)

## Required Context

- The runtime uses curated ModelKind descriptors. Qwen3.5-0.8B and Qwen3.8-27B share an implementation family and operator shapes but retain model-specific dimensions and acceptance records.

## Goal

Consume the validated Milestone 2 `ModelKind` and Qwen descriptor APIs to build an immutable GPU runtime binding for Qwen3.5-0.8B, while proving descriptor-only Qwen3.8-27B dispatch-table construction without weights or execution.

## Non-Goals

- Loading 27B tensors
- Running 27B kernels
- General architecture loading
- Vision descriptors
- MTP execution

## Expected Paths

- read: `docs/PROJECT_STATUS.md`
- read: `docs/ROADMAP.md`
- read: `benchmarks/qwen3.5-0.8b/README.md`
- read: `tools/strix/model.py`
- read: `src/core/model_kind.*`
- read: `models/qwen35_08b/qwen35_descriptor.hpp`
- read: `models/qwen35_08b/qwen35_08b_descriptor.cpp`
- read: `models/qwen38_27b/qwen38_27b_descriptor.*`
- write: `models/qwen35_08b/runtime/model_binding.*`
- write: `tests/models/qwen35_08b/model_binding_test.*`

## Definition of Done

- [ ] Runtime binding consumes, rather than redefines, the Milestone 2 model registry, tensor roles, dimensions, and hybrid block schedule.
- [ ] The 0.8B binding creates explicit dispatch slots for every required text tensor and all 24 blocks.
- [ ] Descriptor-only 27B binding validates dispatch-table dimensions without opening weights, allocating device memory, or claiming kernel promotion.
- [ ] Vision input, unknown kinds, missing roles, duplicate roles, and unsupported encodings fail before GPU initialization.
- [ ] MTP tensors may be recognized for package compatibility but remain absent from the greedy execution graph.

## Development Loop

```sh
nix develop -c cmake --preset hip-test
nix develop -c cmake --build --preset hip-test
git add -A
nix build
nix develop -c ctest --preset hip-test --output-on-failure -R 'qwen35_(08b|27b)_descriptor|checked_shape_arithmetic'
```

## Hardware Validation

```sh
nix develop -c bash -lc './result/bin/strix-server diagnose --gpu-only | grep -F gfx1151'
```

## Output Artifacts

- GPU runtime binding inventory for the 0.8B model.
- Descriptor-only 27B binding validation report.
- Missing/duplicate-role failure fixtures.

## Stop Conditions

- Stop if runtime binding requires copying or redefining Milestone 2 descriptor data instead of consuming its public API.
- Stop if the pinned source revision's config or tensor inventory disagrees with the compiled 0.8B descriptor.
- Stop before any 27B download, allocation, kernel dispatch, or quality claim.

## ROADMAP Traceability

- tasks: M4.2 operation shape contracts
- tasks: M4.5 request-owned model identity
- exitCriteria: Validated model contract prerequisite
- docs: docs/PROJECT_STATUS.md#Models
- docs: docs/ROADMAP.md#Milestone-4:-Qwen-GPU-Only-Vertical-Slice

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
