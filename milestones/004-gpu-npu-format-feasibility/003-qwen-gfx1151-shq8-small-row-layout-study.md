---
id: M004-C003
title: "Benchmark Qwen SHQ8 small-row/prefill candidates and lossless layouts"
milestone: M004
status: planned
dependencies: [M002-C007, M004-C001]
---

# M004-C003: Benchmark Qwen SHQ8 small-row/prefill candidates and lossless layouts

**Track:** gpu

## Dependencies

- [M002-C007](../002-hardware-diagnostics-baselines/007-machine-fingerprint-artifact.md)
- [M004-C001](001-qwen-gfx1151-shq8-decode-gemv.md)

## Required Context

- Not every M greater than one should use WMMA. SHQ8 provides the lower-complexity vehicle for comparing row-tiled GEMV, WMMA-class staging, hipBLASLt where applicable, canonical T16, and a losslessly repacked GPU view.

## Goal

Implement bounded Qwen-private SHQ8 candidates for representative small-row and prefill buckets, then select or reject algorithms and canonical-versus-repacked layouts from correctness-linked end-to-end kernel measurements.

## Non-Goals

- A universal GEMM library
- All possible row counts
- Runtime autotuning on untrusted requests
- Keeping an unaccounted duplicate weight copy
- Full-model prefill

## Expected Paths

- `models/qwen35_08b/gpu/gfx1151/shq8_small_row.hip`
- `models/qwen35_08b/gpu/gfx1151/shq8_layout.cpp`
- `models/qwen35_08b/gpu/gfx1151/shq8_tuning.json`
- `tests/kernels/qwen35_08b/test_shq8_small_row.cpp`
- `tests/kernels/qwen35_08b/test_shq8_repack.cpp`
- `tools/testing/qwen_gfx1151_shq8_small_row_layout_study_bench.cpp`

## Definition of Done

- [ ] At least one row-tiled and one matrix/staging candidate are compared over explicit representative M buckets including 2, 4, 8, 16, 32, 64, 128, and 256 where valid.
- [ ] Every candidate and lossless repack round-trip matches the CPU oracle and canonical codes/scales/zero points; padding and active rows cannot affect logical output.
- [ ] Report compares canonical T16 against a GPU-native lossless view including one-time repack cost, duplicate resident bytes, alignment, ISA, occupancy, bandwidth, launch count, and sustained performance.
- [ ] Selection table retains only measured shape/algorithm/layout choices; unsupported shapes fall back to a correct eager path.
- [ ] No timed-path allocation or load-time duplicate is hidden from accounting.

## Development Loop

```sh
mkdir -p artifacts/m004
git add -A
nix build
nix develop -c cmake -S . -B build-m004-c003 -DENGINE_ENABLE_HIP=ON -DENGINE_ENABLE_XRT=OFF -DGPU_TARGETS=gfx1151 -DBUILD_TESTING=ON
nix develop -c cmake --build build-m004-c003
STRIX_ARTIFACT_DIR=artifacts/m004 STRIX_REQUIRE_GFX1151=1 nix develop -c ctest --preset hip-test --output-on-failure -R 'm004_c003_hardware'
nix develop -c ctest --test-dir build-m004-c003 --output-on-failure -R 'qwen35_shq8_(small_row|repack)'
```

## Hardware Validation

- gfx1151 is required for candidate selection and profiler evidence. Host-only tests can validate repack losslessness, but cannot select the GPU layout or threshold.

## Output Artifacts

- artifacts/m004/c003-gfx1151.json
- models/qwen35_08b/gpu/gfx1151/shq8_tuning.json
- canonical/repacked checksum map and byte-accounting report

## Stop Conditions

- Reject any repack that changes numerical values or lacks explicit duplicate-memory accounting.
- Do not promote WMMA solely because M is greater than one.
- Stop benchmarking a candidate after correctness failure; retain the failure record.
- Do not generalize Qwen3.5 results into Qwen3.8 tuning without a separate acceptance record.

## ROADMAP Traceability

- docs/ROADMAP.md#milestone-3-early-gpu-and-npu-format-feasibility — GPU track items 3–5
- docs/GPU_BACKEND.md#prefill-and-batched-verification
- docs/HETEROGENEOUS_EXECUTION.md#shared-weights
- docs/MEMORY_MANAGEMENT.md#tier-b-shared-source-with-backend-views

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
