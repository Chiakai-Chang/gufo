---
id: M004-C011
title: "Close the blocking gfx1151 format-feasibility gate"
milestone: M004
status: planned
dependencies: [M004-C001, M004-C002, M004-C003, M004-C004]
---

# M004-C011: Close the blocking gfx1151 format-feasibility gate

**Track:** GPU blocking gate

## Dependencies

- [M004-C001](001-qwen-gfx1151-shq8-decode-gemv.md)
- [M004-C002](002-qwen-gfx1151-shq4-decode-gemv.md)
- [M004-C003](003-qwen-gfx1151-shq8-small-row-layout-study.md)
- [M004-C004](004-qwen-gfx1151-shq4-small-row-layout-study.md)

## Required Context

- `docs/ROADMAP.md`, Milestone 3 GPU track and exit criteria.
- `docs/GPU_BACKEND.md`.
- `docs/QUANTIZATION.md` candidate SHQ contract.
- GPU reports produced by M004-C001 through M004-C004.
- Machine fingerprint contract from M002-C007.

## Goal

Produce the machine-checkable GPU-only Milestone 3 disposition that selects the
usable gfx1151 layouts and kernel identities required by Milestone 4 without
waiting for NPU or shared-allocation results.

## Non-Goals

- NPU or heterogeneous route promotion.
- SHQ v1 freezing.
- Full-model inference.
- Re-running or retuning kernels inside the gate card.
- Choosing a production Qwen3.8-27B recipe.

## Expected Paths

- Read: `artifacts/m004/c001-*` through `artifacts/m004/c004-*`.
- Add: `tests/contracts/m004_gpu_gate_test.*`.
- Add: `tools/testing/m004_gpu_gate.*`.
- Produce: `artifacts/m004/gpu-format-gate.json`.
- Update: `docs/generated/m004-gpu-format-summary.md`.

## Definition of Done

- [ ] CPU and HIP comparisons pass their declared numerical contracts for every promoted SHQ4 and SHQ8 decode shape.
- [ ] Selected common or losslessly repacked layouts are named per tensor/shape family with memory costs and content relationships.
- [ ] ISA, occupancy, measured bandwidth, alignment, logical-tail behavior, and raw repetitions are linked to the M002 machine fingerprint.
- [ ] Slower or unstable candidates are explicitly disabled rather than omitted from the report.
- [ ] The report states that SHQ remains a candidate contract and makes no AIE or portable-v1 claim.
- [ ] The output schema can be consumed by M005-C001 without reading prose or NPU artifacts.

## Development Loop

```sh
nix develop -c cmake --preset hip-test
nix develop -c cmake --build --preset hip-test
nix develop -c ctest --preset hip-test --output-on-failure -R 'm004_gpu_gate'
mkdir -p artifacts/m004 docs/generated
nix develop -c python3 tools/testing/m004_gpu_gate.py \
  --inputs artifacts/m004 \
  --out artifacts/m004/gpu-format-gate.json \
  --markdown docs/generated/m004-gpu-format-summary.md
```

## Hardware Validation

The gate consumes retained gfx1151 reports from its dependencies. If any report
was captured on another architecture, lacks raw repetitions, or lacks a machine
fingerprint, the gate fails instead of silently substituting a result.

## Output Artifacts

- `artifacts/m004/gpu-format-gate.json`.
- `docs/generated/m004-gpu-format-summary.md`.
- Selected kernel/layout compatibility identifiers for Milestone 4.

## Stop Conditions

- Stop if any promoted result lacks a CPU comparison or exact machine identity.
- Stop if closing the GPU gate would require NPU, dma-buf, or concurrency evidence.
- Stop if a format/layout change invalidates the committed conformance vectors; open a Milestone 2 contract revision instead.

## ROADMAP Traceability

- Milestone 3 GPU track tasks 1–5.
- Exit: CPU and HIP agree under their numerical contracts.
- Exit: the common layout has measured GPU evidence.
- Exit: unstable or slower NPU paths do not block GPU work.

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes.
- Rejected kernel/layout candidates and reasons.
- Residual risks without expanding this card's scope.
