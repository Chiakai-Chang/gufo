---
id: M004-C008
title: "Implement AIE2P BF16 GEMM and quantify reusable shape-bucket policy"
milestone: M004
status: planned
dependencies: [M003-C008, M004-C005]
---

# M004-C008: Implement AIE2P BF16 GEMM and quantify reusable shape-bucket policy

**Track:** npu

## Dependencies

- [M003-C008](../003-test-oracles-model-contracts/008-high-precision-cpu-operator-oracles.md)
- [M004-C005](005-qwen-aie2p-xrt-program-lifecycle.md)

## Required Context

- BF16 is the bring-up/reference path for sensitive tensors, while public evidence shows configuration can cost roughly as much as a large GEMM. A bounded reusable bucket policy must be measured rather than inferred.

## Goal

Implement a Qwen-private BF16 AIE2P GEMM reference family and use it to quantify shape padding, assigned-column variants, configuration reuse, DMA overlap, and batch-1 GEMV separately.

## Non-Goals

- Full transformer operators
- One program per request shape
- NPU batch-1 decode ownership
- A universal AIE catalog
- GPU/NPU concurrency

## Expected Paths

- `models/qwen35_08b/npu/aie2p/bf16/`
- `models/qwen35_08b/npu/aie2p/bf16/metadata.json`
- `models/qwen35_08b/npu/aie2p/shape_buckets.json`
- `tests/kernels/qwen35_08b/test_aie2p_bf16.cpp`
- `tools/testing/qwen_aie2p_bf16_shape_buckets_bench.cpp`

## Definition of Done

- [ ] BF16 output matches the independent CPU reference under declared BF16 storage and accumulation semantics; reports do not call BF16 FP32.
- [ ] Physical M buckets at a small declared set are tested with logical masks, padding cost, and multiple actually assigned column counts.
- [ ] Context/program reuse is compared with reload/reconfiguration, and serialized transfer/compute is compared with overlapped DMA/compute.
- [ ] M=1 GEMV and large GEMM have separate latency/throughput series with dispatch and completion overhead included.
- [ ] A checked-in model-local bucket metadata file contains only evidence-supported combinations and an explicit GPU fallback for unpromoted shapes.

## Development Loop

```sh
mkdir -p artifacts/m004
git add -A
nix build
nix develop -c cmake -S . -B build-m004-c008 -DENGINE_ENABLE_HIP=OFF -DENGINE_ENABLE_XRT=ON -DBUILD_TESTING=ON
nix develop -c cmake --build build-m004-c008
STRIX_ARTIFACT_DIR=artifacts/m004 STRIX_REQUIRE_XDNA2=1 nix develop -c ctest --preset npu-test --output-on-failure -R 'm004_c008_hardware'
nix develop -c ctest --test-dir build-m004-c008 --output-on-failure -R qwen35_aie2p_bf16
```

## Hardware Validation

- Requires XDNA2/AIE2P and compatible pinned software/firmware for acceptance. Tests should also exercise reduced assigned columns where the resource solver permits. NPU unavailability is non-blocking for GPU MVP.

## Output Artifacts

- artifacts/m004/c008-xdna2.json
- models/qwen35_08b/npu/aie2p/shape_buckets.json
- BF16 CPU/AIE correctness and lifecycle-cost report

## Stop Conditions

- Reject a bucket whose padding/configuration/DMA costs erase usefulness.
- Stop on timeout or firmware reset and quarantine the context.
- Do not infer M=1 performance from GEMM TOPS.
- Do not promote NPU merely to keep it occupied.

## ROADMAP Traceability

- docs/ROADMAP.md#milestone-3-early-gpu-and-npu-format-feasibility — NPU track items 2–3 and 5–6; item 4 belongs to the quantized cards
- docs/NPU_RESEARCH.md#configuration-reuse
- docs/NPU_BACKEND.md#shape-buckets
- docs/TESTING.md#reference-terminology

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
