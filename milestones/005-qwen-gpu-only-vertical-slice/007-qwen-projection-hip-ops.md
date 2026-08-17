---
id: M005-C007
title: "Implement Qwen BF16/SHQ8 projection kernels for gfx1151"
milestone: M005
status: planned
dependencies: [M003-C008, M003-C011, M004-C001, M004-C002, M004-C011, M005-C005]
---

# M005-C007: Implement Qwen BF16/SHQ8 projection kernels for gfx1151

## Dependencies

- [M003-C008](../003-test-oracles-model-contracts/008-high-precision-cpu-operator-oracles.md)
- [M003-C011](../003-test-oracles-model-contracts/011-candidate-shq-byte-vectors.md)
- [M004-C001](../004-gpu-npu-format-feasibility/001-qwen-gfx1151-shq8-decode-gemv.md)
- [M004-C002](../004-gpu-npu-format-feasibility/002-qwen-gfx1151-shq4-decode-gemv.md)
- [M004-C011](../004-gpu-npu-format-feasibility/011-gpu-format-feasibility-gate.md)
- [M005-C005](005-request-state-and-gpu-arenas.md)

## Required Context

- Bring-up should avoid debugging every low-bit path. A model-private BF16 or SHQ8 eager projection path is sufficient to unlock the full model; SHQ4 follows after the greedy slice.

## Goal

Implement decode GEMV and small-row/prefill projection dispatch consuming final artifact bytes for Q/K/V, attention output, DeltaNet projections, FFN gate/up/down, and LM-head-compatible matrix shapes.

## Non-Goals

- SHQ4 acceptance
- Runtime repacking
- NPU kernels
- Generic shared GEMM registry
- Performance promotion from microbenchmarks alone

## Expected Paths

- read: `docs/GPU_BACKEND.md`
- read: `docs/QUANTIZATION.md`
- read: `tools/strix/shq.py`
- write: `models/qwen35_08b/gpu/gfx1151/projection*.hip`
- write: `models/qwen35_08b/cpu/oracles/projection.*`
- write: `tests/kernels/qwen35_projection_test.*`

## Definition of Done

- [ ] Decode and retained prefill variants pass CPU dequantization/oracle tests for every 0.8B projection shape, alignment, and tail.
- [ ] Packed SHQ8 is consumed directly with no timed or load-time numerical repack; BF16 fallback is explicit.
- [ ] Dispatch rejects unsupported encoding/shape combinations before launch.
- [ ] Other model code-object hashes are unchanged.

## Development Loop

```sh
mkdir -p artifacts/m005
nix develop -c cmake --preset hip-test
nix develop -c cmake --build --preset hip-test
git add -A
nix build
nix develop -c ctest --preset hip-test --output-on-failure -R 'qwen35_(projection_cpu|dispatch)'
```

## Hardware Validation

```sh
STRIX_ARTIFACT_DIR=artifacts/m005 STRIX_REQUIRE_GFX1151=1 nix develop -c ctest --preset hip-test --output-on-failure -R 'm005_c007_hardware'
```

## Output Artifacts

- Per-shape CPU/HIP error matrix
- gfx1151 ISA, occupancy, alignment, and bandwidth report

## Stop Conditions

- Stop if packed bytes require an unaccounted runtime repack.
- Stop if any projection is silently routed to an unsupported encoding.

## ROADMAP Traceability

- tasks: M4.2 Q/K/V and output, gate/up/down, LM-head projection primitives
- exitCriteria: Layer outputs and full logits prerequisite
- docs: docs/GPU_BACKEND.md#Quantization-Paths
- docs: docs/ROADMAP.md#Milestone-4:-Qwen-GPU-Only-Vertical-Slice

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
