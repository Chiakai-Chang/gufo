---
id: M004-C006
title: "Implement AIE2P W8A8 SHQ8-T16 GEMM and independent GEMV measurements"
milestone: M004
status: planned
dependencies: [M003-C008, M003-C011, M004-C005]
---

# M004-C006: Implement AIE2P W8A8 SHQ8-T16 GEMM and independent GEMV measurements

**Track:** npu

## Dependencies

- [M003-C008](../003-test-oracles-model-contracts/008-high-precision-cpu-operator-oracles.md)
- [M003-C011](../003-test-oracles-model-contracts/011-candidate-shq-byte-vectors.md)
- [M004-C005](005-qwen-aie2p-xrt-program-lifecycle.md)

## Required Context

- Public NPU TOPS evidence is dominated by large GEMM and cannot establish batch-1 decode. SHQ8 must be tested on reusable physical buckets, with canonical T16 versus lossless backend views and all DMA/configuration costs visible.

## Goal

Implement a Qwen-private AIE2P W8A8 kernel family for a bounded SHQ8-T16 shape matrix, validate its scale/layout contract, and report large GEMM separately from batch-1 GEMV.

## Non-Goals

- Routing batch-1 decode to NPU
- Full-model NPU prefill
- Per-request JIT/reconfiguration
- Hidden backend copies
- Production promotion from TOPS alone

## Expected Paths

- `models/qwen35_08b/npu/aie2p/w8a8/`
- `models/qwen35_08b/npu/aie2p/w8a8/metadata.json`
- `tests/kernels/qwen35_08b/test_aie2p_w8a8.cpp`
- `tools/testing/qwen_aie2p_w8a8_shq8_matrix_bench.cpp`

## Definition of Done

- [ ] Canonical candidate SHQ8 T16 ordering and scale epilogue match the independent CPU oracle across typical, padded, masked, and tail shapes.
- [ ] A bounded reusable M bucket set records logical/physical rows and requested/assigned columns; underfilled buckets report padding cost and GPU fallback guidance.
- [ ] Large/high-intensity GEMM and M=1 GEMV are separate series; no GEMM TOPS value is presented as decode performance.
- [ ] Canonical row/T16 and any losslessly repacked/column-oriented view are compared with repack time, duplicate bytes, DMA, submission, synchronization, and completion included.
- [ ] Serialized versus overlapped DMA/compute measurements and program reuse versus reconfiguration measurements are retained.

## Development Loop

```sh
mkdir -p artifacts/m004
git add -A
nix build
nix develop -c cmake -S . -B build-m004-c006 -DENGINE_ENABLE_HIP=OFF -DENGINE_ENABLE_XRT=ON -DBUILD_TESTING=ON
nix develop -c cmake --build build-m004-c006
STRIX_ARTIFACT_DIR=artifacts/m004 STRIX_REQUIRE_XDNA2=1 nix develop -c ctest --preset npu-test --output-on-failure -R 'm004_c006_hardware'
nix develop -c ctest --test-dir build-m004-c006 --output-on-failure -R qwen35_aie2p_w8a8
```

## Hardware Validation

- Numerical and timing acceptance requires compatible XDNA2/AIE2P, pinned XRT/amdxdna/firmware, and pinned AIE compiler. GPU presence is not required for this card. Unsupported NPU closes with evidence and does not block C001-C004 or Milestone 4 GPU work.

## Output Artifacts

- artifacts/m004/c006-xdna2.json
- embedded program and metadata content hashes
- W8A8 CPU/AIE comparison matrix and layout checksum map

## Stop Conditions

- Stop benchmarking on any numerical mismatch or non-finite output.
- Disable rather than promote if program/configuration or DMA cost removes the declared benefit.
- Do not use M=1 data inferred from M greater than one.
- Do not interpret success as SHQ v1 readiness without every other gate.

## ROADMAP Traceability

- docs/ROADMAP.md#milestone-3-early-gpu-and-npu-format-feasibility — NPU track items 2–6
- docs/NPU_RESEARCH.md#required-probe-matrix
- docs/NPU_BACKEND.md#shape-buckets
- docs/TESTING.md#t2-device-kernel-tests

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
