---
id: M002-C004
title: "Add deterministic HIP allocation, copy, kernel, event, and graph smoke tests"
milestone: M002
status: planned
dependencies: [M002-C007, M001-C005, M001-C008]
---

# M002-C004: Add deterministic HIP allocation, copy, kernel, event, and graph smoke tests

## Dependencies

- [M002-C007](007-machine-fingerprint-artifact.md)
- [M001-C005](../001-repository-toolchain/005-supported-toolchain-pins.md)
- [M001-C008](../001-repository-toolchain/008-single-pr-test-entrypoint.md)

## Required Context

- docs/ROADMAP.md
- docs/GPU_BACKEND.md
- docs/TESTING.md
- docs/PERFORMANCE.md
- src/main.cpp
- CMakeLists.txt
- flake.nix
- .devops/nix/package.nix

## Goal

Implement a bounded HIP smoke sequence on gfx1151 that repeatedly validates allocation, bidirectional copies, deterministic kernel execution, event ordering/timing, and eager-versus-graph replay equivalence.

## Non-Goals

- Performance tuning
- Model numerical kernels
- Treating launch success as numerical correctness
- Supporting CUDA or non-gfx1151 production architectures
- Using HIP graphs before eager output has been checked

## Expected Paths

- `src/core/hip/smoke.hip`
- `src/core/hip/smoke.h`
- `src/core/diagnostics/hip_smoke.cpp`
- `tests/device/hip_smoke_test.cpp`
- `tests/fixtures/diagnostics/hip_smoke_schema.json`
- `docs/CLI.md`
- `CMakeLists.txt`

## Definition of Done

- [ ] The test allocates source/destination/device buffers, uses deterministic seeded inputs, verifies H2D and D2H bytes, launches a nontrivial deterministic kernel, and checks every output element against a CPU-computed expectation
- [ ] Events establish stream ordering and produce finite nonnegative elapsed time
- [ ] A graph captures and replays the already-passing eager sequence; eager and every replay are byte-identical
- [ ] At least 100 repeated eager and 100 repeated graph executions complete without mismatch, leaked allocation, stale error, or growth
- [ ] Wrong architecture, zero devices, allocation failure, launch failure, timeout, and mismatch produce actionable nonzero diagnostics
- [ ] The JSON result includes fingerprint ID, architecture, iteration counts, first mismatch, HIP error name, and per-stage status
- [ ] Device tests are labeled hardware-required and do not become false green skips
- [ ] One focused PR touches only HIP smoke implementation/tests and diagnose integration

## Development Loop

```sh
nix develop -c cmake --preset test
nix develop -c cmake --build --preset test
git add -A
nix build
nix flake check
./result/bin/strix-server diagnose --smoke hip --iterations 100 --graph-iterations 100 --timeout-ms 30000 --json --output /tmp/strix-hip-smoke.json
./result/bin/strix-server diagnose --validate-artifact /tmp/strix-hip-smoke.json
nix develop -c ctest --preset test --output-on-failure -L hip-hardware
# Run on gfx1151 with render-group access. Any requested hardware test skip is a failed acceptance run.
```

## Output Artifacts

- /tmp/strix-hip-smoke.json (hardware attestation retained externally)
- tests/fixtures/diagnostics/hip_smoke_schema.json

## Stop Conditions

- Detected GPU architecture is not exactly gfx1151
- HIP runtime or device permissions are unavailable
- Graph support is unavailable on the pinned supported ROCm stack
- Any iteration has a numerical mismatch, timeout, reset, or allocation growth
- Validation would rely only on hipGetLastError without checking output bytes

## ROADMAP Traceability

- docs/ROADMAP.md Milestone 1 task 4
- docs/ROADMAP.md Milestone 1 exit criterion 2
- docs/GPU_BACKEND.md#streams-and-events
- docs/GPU_BACKEND.md#hip-graphs
- docs/TESTING.md#t2-device-kernel-tests

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
