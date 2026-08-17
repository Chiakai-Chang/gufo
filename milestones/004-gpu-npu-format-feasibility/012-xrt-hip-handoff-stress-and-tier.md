---
id: M004-C012
title: "Stress explicit XRT/HIP handoffs and select the interoperability tier"
milestone: M004
status: planned
dependencies: [M004-C003, M004-C004, M004-C009]
---

# M004-C012: Stress explicit XRT/HIP handoffs and select the interoperability tier

**Track:** shared-allocation, non-blocking research

## Dependencies

- [M004-C003](003-qwen-gfx1151-shq8-small-row-layout-study.md)
- [M004-C004](004-qwen-gfx1151-shq4-small-row-layout-study.md)
- [M004-C009](009-xrt-bo-hip-explicit-handoff-probe.md)

## Required Context

- `docs/NPU_RESEARCH.md`, ordering, visibility, and required probe matrix.
- `docs/MEMORY_MANAGEMENT.md`, interoperability tiers.
- `docs/HETEROGENEOUS_EXECUTION.md`, synchronization and promotion rules.
- Basic supported-positive or supported-negative record from M004-C009.
- Machine fingerprint from M002-C007.

## Goal

For platforms where basic import works, stress explicit bidirectional ownership
handoffs across sizes and partial ranges, measure each transition component, and
select Tier A, B, C, or D from retained evidence. When import is unsupported,
close with a measured fallback-tier decision without blocking the GPU MVP.

## Non-Goals

- Concurrent writes of any kind.
- Implicit dma-buf fencing.
- Sustained independent GPU/NPU workload concurrency; M004-C010 owns it.
- Model-layer routing or production scheduler integration.
- SHQ format freezing.

## Expected Paths

- Read: `artifacts/m004/c009-basic-import.json`.
- Read: `artifacts/m004/c003-gfx1151.json` and `artifacts/m004/c004-gfx1151.json`.
- Add: `tools/testing/xrt_hip_handoff_stress.*`.
- Add: `tests/integration/xrt_hip_handoff_stress_test.*`.
- Add: `CMakeLists.txt`.
- Produce: `artifacts/m004/c012-handoff-stress.json`.
- Produce: `artifacts/m004/interoperability-tier.json`.

## Definition of Done

- [ ] Supported import is tested at 4 KiB, several MiB, and at least 64 MiB with full and partial aligned ranges for at least 10,000 alternating handoffs.
- [ ] HIP-write/NPU-read and NPU-write/HIP-read use explicit producer completion, required visibility maintenance, and consumer submission.
- [ ] Timing separates HIP completion, XRT completion, BO/cache maintenance, zero-copy ownership transition, and explicit-copy fallback.
- [ ] Candidate canonical bytes are converted to each evidence-supported lossless GPU/NPU backend view and round-tripped byte-/value-exact across group, tile, alignment, padded-shape, and tail boundaries; retained reports include view hashes, duplicate bytes, and conversion cost.
- [ ] When direct import or a lossless view is unavailable, an explicit-copy fallback is tested end-to-end and its ownership, synchronization, extra memory, and latency determine Tier C/D rather than being inferred.
- [ ] Immutable simultaneous reads are tested only after producer completion; all concurrent writes remain rejected by policy tests.
- [ ] Corruption, bounded-wait timeout, reset, stale mapping, FD leak, and cleanup failures terminate the run and retain the failing iteration.
- [ ] The final record selects Tier A, B, C, or D with exact reasons and leaves GPU-only operation available.
- [ ] A supported-negative M004-C009 result closes this card with Tier C/D fallback evidence and does not fabricate stress results.

## Development Loop

```sh
nix develop -c cmake --preset npu-test
nix develop -c cmake --build --preset npu-test
nix develop -c ctest --preset npu-test --output-on-failure -R 'xrt_hip_handoff_policy'
mkdir -p artifacts/m004
nix develop -c ./build/npu-test/xrt-hip-handoff-stress \
  --iterations 10000 \
  --sizes 4096,4194304,67108864 \
  --partial-ranges \
  --explicit-ordering \
  --view-report artifacts/m004/c012-backend-view-conformance.json \
  --out artifacts/m004/c012-handoff-stress.json
```

## Hardware Validation

Requires gfx1151 and XDNA2 only for the supported-positive stress path. Import
incompatibility or a driver failure is a valid evidence outcome when captured
with exact software/firmware identity; it must never change GPU-only readiness.

## Output Artifacts

- `artifacts/m004/c012-handoff-stress.json`.
- `artifacts/m004/interoperability-tier.json`.
- `artifacts/m004/c012-backend-view-conformance.json` with round-trip hashes and fallback costs.
- Transition-cost distributions and failing-iteration record when applicable.

## Stop Conditions

- Stop immediately on corruption, timeout, firmware stall, reset, unsafe thermal state, or stale mapping.
- Do not convert a failed stress run into Tier A by excluding failed iterations.
- Do not attempt concurrent writes or implicit-fence experiments.
- Do not block M004-C011 or any Milestone 4 GPU dependency.

## ROADMAP Traceability

- Milestone 3 shared-allocation tasks 2–5 and 7–8.
- Exit: memory interoperability tier selected from evidence.
- Exit: unstable NPU/interop paths remain disabled instead of blocking GPU work.
- `docs/NPU_RESEARCH.md#required-probe-matrix`.

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands and results.
- Selected tier and evidence hashes.
- Any failure iteration, reset, timeout, or corruption evidence.
- Residual risks without expanding this card's scope.
