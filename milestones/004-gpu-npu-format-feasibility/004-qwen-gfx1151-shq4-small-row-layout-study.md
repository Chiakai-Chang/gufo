---
id: M004-C004
title: "Benchmark Qwen SHQ4 small-row/prefill candidates and lossless layouts"
milestone: M004
status: planned
dependencies: [M002-C007, M004-C002]
---

# M004-C004: Benchmark Qwen SHQ4 small-row/prefill candidates and lossless layouts

**Track:** gpu

## Dependencies

- [M002-C007](../002-hardware-diagnostics-baselines/007-machine-fingerprint-artifact.md)
- [M004-C002](002-qwen-gfx1151-shq4-decode-gemv.md)

## Required Context

- Q4 prefill must include activation handling, fused scale/zero epilogues, and the actual cost of staging T16 weights. A GPU-native view is permissible only as a measured, lossless, explicitly budgeted derivative of canonical bytes.

## Goal

Implement bounded Qwen-private SHQ4 small-row/prefill candidates and decide shape algorithms plus canonical-versus-repacked layout from oracle-checked gfx1151 results.

## Non-Goals

- SHQ6 optimization
- Complete Qwen prefill
- Runtime artifact mutation
- A shared kernel across model kinds
- Format freeze

## Expected Paths

- `models/qwen35_08b/gpu/gfx1151/shq4_small_row.hip`
- `models/qwen35_08b/gpu/gfx1151/shq4_layout.cpp`
- `models/qwen35_08b/gpu/gfx1151/shq4_tuning.json`
- `tests/kernels/qwen35_08b/test_shq4_small_row.cpp`
- `tests/kernels/qwen35_08b/test_shq4_repack.cpp`
- `tools/testing/qwen_gfx1151_shq4_small_row_layout_study_bench.cpp`

## Definition of Done

- [ ] Row-tiled and matrix/staging candidates cover a bounded declared M matrix, with activation conversion and all staging/repack time represented in the result.
- [ ] Outputs for canonical and repacked views match the independent CPU dequantization/accumulation contract for boundary, padding, active-row, alignment, and tail cases.
- [ ] Profiler artifact records ISA, occupancy, register/LDS use, effective bandwidth, dispatch count, alignment, tails, thermal steady state, and raw repetitions.
- [ ] A model-local shape table identifies selected algorithm and layout or an explicit fallback; decisions require measured workload improvement, not peak TOPS.
- [ ] A retained repack has a checksum relationship to canonical bytes and explicit creation-time/resident-byte accounting.

## Development Loop

```sh
mkdir -p artifacts/m004
git add -A
nix build
nix develop -c cmake -S . -B build-m004-c004 -DENGINE_ENABLE_HIP=ON -DENGINE_ENABLE_XRT=OFF -DGPU_TARGETS=gfx1151 -DBUILD_TESTING=ON
nix develop -c cmake --build build-m004-c004
STRIX_ARTIFACT_DIR=artifacts/m004 STRIX_REQUIRE_GFX1151=1 nix develop -c ctest --preset hip-test --output-on-failure -R 'm004_c004_hardware'
nix develop -c ctest --test-dir build-m004-c004 --output-on-failure -R 'qwen35_shq4_(small_row|repack)'
```

## Hardware Validation

- gfx1151 and the pinned ROCm stack are mandatory for tuning selection. CPU-only execution validates fixtures/repack only and is never substituted for device evidence.

## Output Artifacts

- artifacts/m004/c004-gfx1151.json
- models/qwen35_08b/gpu/gfx1151/shq4_tuning.json
- lossless repack and memory-cost attestation

## Stop Conditions

- Reject output tolerance invented only to admit a failing kernel.
- Reject any candidate whose apparent speed excludes activation conversion, staging, synchronization, or repack cost.
- Do not retain a duplicate layout without material measured benefit and memory accounting.
- Completion of this GPU card does not imply AIE conformance or SHQ v1 readiness.

## ROADMAP Traceability

- docs/ROADMAP.md#milestone-3-early-gpu-and-npu-format-feasibility — GPU track items 3–5
- docs/GPU_BACKEND.md#quantization-paths
- docs/TESTING.md#performance-method
- docs/PROJECT_STATUS.md#shq-format-stability

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
