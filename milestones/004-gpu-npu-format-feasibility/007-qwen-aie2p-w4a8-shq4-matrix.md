---
id: M004-C007
title: "Implement AIE2P W4A8 SHQ4-T16 GEMM and independent GEMV measurements"
milestone: M004
status: planned
dependencies: [M003-C008, M003-C011, M004-C005]
---

# M004-C007: Implement AIE2P W4A8 SHQ4-T16 GEMM and independent GEMV measurements

**Track:** npu

## Dependencies

- [M003-C008](../003-test-oracles-model-contracts/008-high-precision-cpu-operator-oracles.md)
- [M003-C011](../003-test-oracles-model-contracts/011-candidate-shq-byte-vectors.md)
- [M004-C005](005-qwen-aie2p-xrt-program-lifecycle.md)

## Required Context

- The first custom NPU quantized path requires separate INT8 activations and INT4/UINT4 weights, native 4x16x16 AIE2P microtiles, INT32 partials, and fused zero/activation/weight-scale epilogues.

## Goal

Implement and validate a Qwen-private W4A8 AIE2P family over a small reusable shape matrix, including exact candidate T16 microtile ordering and separate GEMM/GEMV evidence.

## Non-Goals

- Using the ggml shared-dtype GEMM as the production kernel
- NPU decode promotion
- Arbitrary shape compilation
- Concurrent GPU/NPU execution
- Format freeze

## Expected Paths

- `models/qwen35_08b/npu/aie2p/w4a8/`
- `models/qwen35_08b/npu/aie2p/w4a8/metadata.json`
- `tests/kernels/qwen35_08b/test_aie2p_w4a8.cpp`
- `tools/testing/qwen_aie2p_w4a8_shq4_matrix_bench.cpp`

## Definition of Done

- [ ] Dynamic INT8 A rows, SHQ4 U4Z/S4 weight interpretation as declared, INT32 accumulation, zero correction, activation scale, BF16 weight scale, and output conversion match a wide CPU oracle.
- [ ] Tests cover native 4x16x16 microtile ordering, every nibble, group boundaries, padded/tail K and N, active-row masks, and representative reusable M buckets.
- [ ] Large GEMM and batch-1 GEMV results are explicitly separated and keyed by assigned AIE columns.
- [ ] Report includes activation packing, DMA, configuration/program reuse, command, completion, padding, and any lossless backend-view costs, plus serialized versus overlapped DMA/compute.
- [ ] Artifact metadata binds architecture, compiler, Qwen implementation, tensor contract, shape buckets, and content hashes.

## Development Loop

```sh
mkdir -p artifacts/m004
git add -A
nix build
nix develop -c cmake -S . -B build-m004-c007 -DENGINE_ENABLE_HIP=OFF -DENGINE_ENABLE_XRT=ON -DBUILD_TESTING=ON
nix develop -c cmake --build build-m004-c007
STRIX_ARTIFACT_DIR=artifacts/m004 STRIX_REQUIRE_XDNA2=1 nix develop -c ctest --preset npu-test --output-on-failure -R 'm004_c007_hardware'
nix develop -c ctest --test-dir build-m004-c007 --output-on-failure -R qwen35_aie2p_w4a8
```

## Hardware Validation

- Requires Strix Halo XDNA2/AIE2P and pinned AIE/XRT/amdxdna/firmware for acceptance. Missing mixed W4A8 support or unstable execution selects disabled-NPU evidence and cannot block the GPU MVP.

## Output Artifacts

- artifacts/m004/c007-xdna2.json
- embedded W4A8 program/metadata hashes
- microtile/epilogue conformance report

## Stop Conditions

- Stop on the first unexplained CPU/AIE mismatch; retain the smallest failing tile.
- Do not hide activation quantization, padding, repacking, configuration, or DMA cost.
- Do not route decode to NPU from GEMM peak results.
- Do not freeze SHQ if this card is unavailable, failing, or not promoted.

## ROADMAP Traceability

- docs/ROADMAP.md#milestone-3-early-gpu-and-npu-format-feasibility — NPU track items 2–6
- docs/NPU_BACKEND.md#first-kernel-family
- docs/NPU_RESEARCH.md#arithmetic-intensity
- docs/PROJECT_STATUS.md#promotion-gates

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
