---
id: M004-C009
title: "Prove or reject basic XRT-BO import into HIP external memory"
milestone: M004
status: planned
dependencies: [M002-C004, M002-C005, M002-C007, M004-C005]
---

# M004-C009: Prove or reject basic XRT-BO import into HIP external memory

**Track:** shared-allocation

## Dependencies

- [M002-C004](../002-hardware-diagnostics-baselines/004-hip-deterministic-smoke.md)
- [M002-C005](../002-hardware-diagnostics-baselines/005-xrt-deterministic-smoke.md)
- [M002-C007](../002-hardware-diagnostics-baselines/007-machine-fingerprint-artifact.md)
- [M004-C005](005-qwen-aie2p-xrt-program-lifecycle.md)

## Required Context

- XRT BO export and HIP external-memory import APIs exist, but the exact amdxdna/XRT PRIME FD to gfx1151 path is not a supported contract until measured. Ordering and cache visibility must be explicit; implicit dma-buf fencing and concurrent writes are excluded.

## Goal

Build an isolated version-checked probe that exports one XRT-owned BO, imports and maps it through HIP, verifies basic GPU/NPU visibility with explicit serialized waits, and records a supported-positive or supported-negative capability result.

## Non-Goals

- Implicit cross-driver fencing
- Multi-size or long-running stress; M004-C012 owns it.
- Handoff latency or fallback-tier selection; M004-C012 owns it.
- Same-range or disjoint concurrent writes.
- Per-layer device ping-pong.
- Assuming unified physical memory implies pointer visibility
- Making NPU/interop a GPU readiness requirement

## Expected Paths

- `src/core/memory/allocation_broker.hpp`
- `src/core/memory/shared_allocation_probe.cpp`
- `src/core/xdna2/linux_export_fd.cpp`
- `src/core/hip/external_memory.cpp`
- `tests/integration/test_shared_allocation.cpp`
- `tests/fixtures/interop/basic_patterns.*`
- `CMakeLists.txt`

## Definition of Done

- [ ] Probe queries hipDeviceAttributeDmaBufSupported, allocates an XRT BO, isolates version-checked PRIME-FD extraction, imports/maps it through HIP, and validates cleanup/error paths without stale mappings.
- [ ] HIP-write/NPU-read and NPU-write/HIP-read use producer completion, required BO/cache visibility maintenance, then consumer submission; xrt::bo::sync is never treated as execution ordering.
- [ ] One 4 KiB BO completes XRT export, HIP import/map, HIP-write/NPU-read, and NPU-write/HIP-read with explicit serialized completion and documented cache maintenance.
- [ ] The adapter reports the exact XRT, HIP, driver, firmware, handle type, ownership, FD-lifetime, and cleanup behavior.
- [ ] Unsupported import, permission denial, or mapping incompatibility produces a bounded structured negative result rather than a crash or GPU-MVP failure.
- [ ] Cleanup tests prove no stale mapping, leaked FD, BO, HIP external-memory object, or device context after success and injected failures.
- [ ] No implicit-fence, concurrent-access, performance, or Tier-A claim is made.

## Development Loop

```sh
git add -A
nix build
nix develop -c cmake -S . -B build-m004-c009 -DENGINE_ENABLE_HIP=ON -DENGINE_ENABLE_XRT=ON -DGPU_TARGETS=gfx1151 -DBUILD_TESTING=ON
nix develop -c cmake --build build-m004-c009 --target strix-test-shared-allocation
nix develop -c ctest --test-dir build-m004-c009 --output-on-failure -R shared_allocation_basic
mkdir -p artifacts/m004
nix develop -c ./build-m004-c009/strix-test-shared-allocation --size 4096 --serialized --json artifacts/m004/c009-basic-import.json
```

## Hardware Validation

- Requires one Strix Halo host exposing gfx1151 and XDNA2 with pinned ROCm, XRT, `amdxdna`, firmware, and dma-buf permissions. Unsupported import is a valid completed research result and does not fail the GPU MVP.

## Output Artifacts

- `artifacts/m004/c009-basic-import.json`.
- Version-checked Linux FD adapter capability record.
- Import/map and cleanup test report.

## Stop Conditions

- Immediately stop and preserve evidence on corruption, stale mapping, driver reset, or bounded-wait timeout.
- Never test concurrent writes in this card; assert they are rejected.
- Do not label successful FD import as Tier A; M004-C012 owns stress, cost measurement, and fallback-tier selection.
- On failure select fallback/disable NPU and keep GPU-only readiness; do not block GPU MVP.

## ROADMAP Traceability

- docs/ROADMAP.md#milestone-3-early-gpu-and-npu-format-feasibility — shared-allocation item 1 only; M004-C012 owns items 2–5
- docs/NPU_RESEARCH.md#required-probe-matrix
- docs/HETEROGENEOUS_EXECUTION.md#shared-memory-and-synchronization
- docs/MEMORY_MANAGEMENT.md#interoperability-tiers

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
