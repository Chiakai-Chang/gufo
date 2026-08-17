---
id: M004-C001
title: "Implement and characterize Qwen-private SHQ8-T16 decode GEMV"
milestone: M004
status: planned
dependencies: [M001-C008, M002-C007, M003-C008, M003-C011, M003-C013]
---

# M004-C001: Implement and characterize Qwen-private SHQ8-T16 decode GEMV

**Track:** gpu

## Dependencies

- [M001-C008](../001-repository-toolchain/008-single-pr-test-entrypoint.md)
- [M002-C007](../002-hardware-diagnostics-baselines/007-machine-fingerprint-artifact.md)
- [M003-C008](../003-test-oracles-model-contracts/008-high-precision-cpu-operator-oracles.md)
- [M003-C011](../003-test-oracles-model-contracts/011-candidate-shq-byte-vectors.md)
- [M003-C013](../003-test-oracles-model-contracts/013-milestone2-contract-gate.md)

## Required Context

- The current native program is only a HIP/XRT probe. The first GPU numerical kernel should use the simpler SHQ8 representation to establish the model-private raw-pointer dispatch, CPU-oracle comparison, and gfx1151 profiling pattern before low-bit complexity is added.

## Goal

Implement one Qwen-owned wave32 batch-1 SHQ8-T16 GEMV path consuming candidate packed planes directly, with deterministic CPU-oracle correctness and retained gfx1151 measurements.

## Non-Goals

- Complete Qwen layer or model inference
- SHQ4, SHQ6, prefill, attention, or sampling
- HIP graphs
- A shared production numerical kernel registry
- Declaring SHQ-T16 v1

## Expected Paths

- `CMakeLists.txt`
- `models/qwen35_08b/gpu/gfx1151/shq8_decode.hip`
- `models/qwen35_08b/gpu/gfx1151/shq8_decode.hpp`
- `tests/kernels/qwen35_08b/test_shq8_decode.cpp`
- `tests/fixtures/formats/shq8_t16/`
- `tools/testing/qwen_gfx1151_shq8_decode_gemv_bench.cpp`

## Definition of Done

- [ ] Kernel accepts raw pointers, fixed-width dimensions, explicit HIP stream, and preallocated scratch; no framework tensor reaches the backend.
- [ ] Candidate SHQ8 planes are consumed without runtime repacking or timed-path allocation.
- [ ] Minimum, typical, non-power-of-two, padded-N/K, alignment-offset, and tail fixtures match the independent CPU oracle under a declared accumulation/tolerance contract; failures report first index, max error, and non-finite count.
- [ ] Generated code is native gfx1151/wave32 and the report records ISA hash, launch geometry, VGPR/SGPR/LDS use, occupancy, effective weight bandwidth, alignment/tail behavior, clocks, warmup, repetitions, and thermal state.
- [ ] Model-local symbol and dependency checks prevent this kernel from becoming shared numerical source.

## Development Loop

```sh
mkdir -p artifacts/m004
git add -A
nix build
nix develop -c cmake -S . -B build-m004-c001 -DENGINE_ENABLE_HIP=ON -DENGINE_ENABLE_XRT=OFF -DGPU_TARGETS=gfx1151 -DBUILD_TESTING=ON
nix develop -c cmake --build build-m004-c001
STRIX_ARTIFACT_DIR=artifacts/m004 STRIX_REQUIRE_GFX1151=1 nix develop -c ctest --preset hip-test --output-on-failure -R 'm004_c001_hardware'
nix develop -c ctest --test-dir build-m004-c001 --output-on-failure -R qwen35_shq8_decode
```

## Hardware Validation

- Build/static checks run on x86_64-linux. Numerical device and performance acceptance requires Strix Halo gfx1151 with the pinned ROCm stack; a missing/wrong GPU is an explicit not-run hardware result, not a pass.

## Output Artifacts

- artifacts/m004/c001-gfx1151.json
- gfx1151 code-object and ISA hash
- CPU/HIP mismatch report or passing matrix

## Stop Conditions

- Stop and preserve a minimized fixture on any numerical mismatch or non-finite output; do not benchmark incorrect output.
- Stop promotion if timed execution allocates or repacks weights.
- Reject builds targeting a gfx1100 alias or containing CUDA dependencies.
- Do not infer full-model speed from this microbenchmark.

## ROADMAP Traceability

- docs/ROADMAP.md#milestone-3-early-gpu-and-npu-format-feasibility — GPU track item 2 and exit numerical agreement
- docs/GPU_BACKEND.md#single-token-decode
- docs/GPU_BACKEND.md#model-isolation
- docs/TESTING.md#t2-device-kernel-tests
- docs/PROJECT_STATUS.md#shq-format-stability

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
