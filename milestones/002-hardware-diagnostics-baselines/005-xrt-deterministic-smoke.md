---
id: M002-C005
title: "Add deterministic XRT discovery, context, buffer, command, and completion smoke tests"
milestone: M002
status: planned
dependencies: [M002-C007, M001-C005, M001-C008]
---

# M002-C005: Add deterministic XRT discovery, context, buffer, command, and completion smoke tests

## Dependencies

- [M002-C007](007-machine-fingerprint-artifact.md)
- [M001-C005](../001-repository-toolchain/005-supported-toolchain-pins.md)
- [M001-C008](../001-repository-toolchain/008-single-pr-test-entrypoint.md)

## Required Context

- docs/ROADMAP.md
- docs/NPU_BACKEND.md
- docs/NPU_RESEARCH.md
- docs/TESTING.md
- docs/PERFORMANCE.md
- src/main.cpp
- CMakeLists.txt
- .devops/nix/xrt.nix
- .devops/nix/xrt-plugin-amdxdna.nix

## Goal

Implement a bounded XRT smoke sequence that discovers XDNA2, creates a workload context, allocates and verifies BOs, submits a reviewed deterministic AIE2P program, and validates finite completion repeatedly.

## Non-Goals

- W4A8/GEMM feasibility kernels
- HIP/XRT shared-buffer import
- Raw ioctl or undocumented HSA production paths
- Throughput promotion
- Continuing after a timeout or reset as if the NPU were healthy

## Expected Paths

- `src/core/xdna2/device.cpp`
- `src/core/xdna2/device.h`
- `src/core/xdna2/smoke.cpp`
- `src/core/xdna2/smoke.h`
- `src/core/xdna2/programs/smoke/`
- `tests/device/xrt_smoke_test.cpp`
- `tests/fixtures/diagnostics/xrt_smoke_schema.json`
- `docs/CLI.md`
- `CMakeLists.txt`
- `.devops/nix/package.nix`

## Definition of Done

- [ ] Discovery identifies an XDNA2/AIE2P device and records assigned resources rather than assuming the physical maximum
- [ ] The runtime creates and destroys a finite-timeout context, allocates/maps/synchronizes XRT BOs, submits a checked-in reviewed smoke program, waits for completion, and verifies deterministic output bytes
- [ ] At least 100 command/completion cycles reuse the context and buffers without mismatch, timeout, stale completion, reset, or allocation growth
- [ ] Executable program bytes are built from pinned reviewed source, content-hashed, architecture/ABI checked, and never loaded from a model artifact
- [ ] Missing plugin/device, incompatible architecture/ABI, unsupported driver/firmware, permission failure, BO failure, command error, and timeout are distinguished with detected/required values and remediation
- [ ] A timeout quarantines the context and makes the command fail; it is never counted as a skip/pass
- [ ] The JSON result includes fingerprint ID, program hash, context/resource details, iteration count, completion status, and first mismatch
- [ ] One focused PR contains only the minimal XRT backend objects needed by this smoke and its tests/integration

## Development Loop

```sh
nix develop -c cmake --preset test
nix develop -c cmake --build --preset test
git add -A
nix build
nix flake check
./result/bin/strix-server diagnose --smoke xrt --iterations 100 --timeout-ms 30000 --json --output /tmp/strix-xrt-smoke.json
./result/bin/strix-server diagnose --validate-artifact /tmp/strix-xrt-smoke.json
nix develop -c ctest --preset test --output-on-failure -L xrt-hardware
# Run with accel-device access and the Nix-packaged XRT amdxdna plugin. A requested test that is skipped is not acceptance evidence.
```

## Output Artifacts

- /tmp/strix-xrt-smoke.json (hardware attestation retained externally)
- Content hash of the embedded AIE2P smoke program in the JSON result
- tests/fixtures/diagnostics/xrt_smoke_schema.json

## Stop Conditions

- No reviewed deterministic AIE2P program can be built reproducibly with the pinned Nix toolchain
- Detected device is not XDNA2/AIE2P
- Driver or firmware is outside the M001 compatibility contract
- Any command times out, resets firmware, mismatches output, or leaks resources
- Implementation requires undocumented raw packets/ioctls without a separately approved architecture decision

## ROADMAP Traceability

- docs/ROADMAP.md Milestone 1 task 5
- docs/ROADMAP.md Milestone 1 exit criteria 2 and 4
- docs/NPU_BACKEND.md#backend-objects
- docs/NPU_BACKEND.md#program-trust
- docs/NPU_BACKEND.md#error-handling

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
