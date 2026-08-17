---
id: M002-C006
title: "Measure first-touch, page-fault, prefault, and thermal steady-state behavior"
milestone: M002
status: planned
dependencies: [M002-C003, M002-C004, M002-C005]
---

# M002-C006: Measure first-touch, page-fault, prefault, and thermal steady-state behavior

## Dependencies

- [M002-C003](003-memory-bandwidth-baselines.md)
- [M002-C004](004-hip-deterministic-smoke.md)
- [M002-C005](005-xrt-deterministic-smoke.md)

## Required Context

- docs/ROADMAP.md
- docs/MEMORY_MANAGEMENT.md
- docs/PERFORMANCE.md
- docs/TESTING.md
- docs/GPU_BACKEND.md
- docs/NPU_BACKEND.md

## Goal

Add a controlled hardware benchmark that separates allocation, first touch, page faults, explicit prefault, warm steady state, and sustained thermal/clock behavior for CPU-, HIP-, and XRT-visible memory and retains a fingerprinted baseline.

## Non-Goals

- Changing OS paging policy or requiring privileged tuning
- GPU/NPU shared-allocation interoperability
- Model loading
- Hiding first-touch work inside timed steady-state results
- Declaring thermal stability from a short burst

## Expected Paths

- `src/core/diagnostics/memory_behavior.cpp`
- `src/core/diagnostics/memory_behavior.h`
- `src/core/diagnostics/thermal_sampler.cpp`
- `src/core/diagnostics/thermal_sampler.h`
- `tests/diagnostics/memory_behavior_test.cpp`
- `tests/fixtures/diagnostics/thermal_schema.json`
- `benchmarks/hardware/README.md`
- `docs/CLI.md`

## Definition of Done

- [ ] The benchmark reports allocation, untouched access, first sequential touch, major/minor page-fault deltas, explicit prefault, warm access, and backend synchronization as separate phases
- [ ] CPU, HIP-visible, and XRT-visible allocation types are named exactly; unavailable counters/sensors are marked with reason
- [ ] A sustained run samples wall time, throughput, CPU/GPU/NPU clocks where exposed, temperatures, power mode, and errors at a documented cadence through warmup and steady state
- [ ] Default acceptance run lasts at least 15 minutes after warmup, retains every sample, and summarizes median/p95 throughput plus start/peak/end temperature and clock throttling
- [ ] The prefault comparison verifies touched bytes and demonstrates that phases were not optimized away
- [ ] Each artifact includes M002-C007 fingerprint ID, command, revision, allocation sizes, page size, NUMA/affinity policy, raw samples, and checksums
- [ ] A sanitized baseline summary is retained in Git and full raw output is retained externally by content hash
- [ ] One focused PR contains memory/fault/thermal measurement and report changes only

## Development Loop

```sh
nix develop -c cmake --preset test
nix develop -c cmake --build --preset test
git add -A
nix build
nix flake check
./result/bin/strix-server diagnose --benchmark memory-behavior --backends cpu,hip,xrt --working-set-mib 4096 --warmup-seconds 120 --duration-seconds 900 --sample-ms 1000 --json --output /tmp/strix-memory-thermal.json
./result/bin/strix-server diagnose --validate-artifact /tmp/strix-memory-thermal.json
nix develop -c ctest --preset test --output-on-failure -R 'memory_behavior|thermal_schema'
# Run on AC power under the documented production power mode with unrelated workloads stopped. If 4 GiB is unsafe for the host, stop and document capacity rather than silently reducing the working set.
```

## Output Artifacts

- /tmp/strix-memory-thermal.json (full 15-minute raw baseline retained externally)
- benchmarks/hardware/README.md (sanitized thermal and page-fault baseline summary, checksum, fingerprint ID)
- tests/fixtures/diagnostics/thermal_schema.json

## Stop Conditions

- Required temperature/clock sampling is unavailable and no authoritative source can be identified
- Memory pressure risks OOM or system instability
- Any HIP/XRT timeout, reset, corruption, or sentinel mismatch occurs
- The benchmark is interrupted, thermally nonstationary, or run under a different power mode than recorded
- Raw samples, phase boundaries, or fingerprint ID are missing

## ROADMAP Traceability

- docs/ROADMAP.md Milestone 1 task 6
- docs/ROADMAP.md Milestone 1 exit criterion 3
- docs/PERFORMANCE.md#measurement-rules
- docs/PERFORMANCE.md#promotion-gates
- docs/TESTING.md#performance-method

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
