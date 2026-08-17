---
id: M002-C003
title: "Measure sustained CPU, GPU, and NPU-visible memory bandwidth"
milestone: M002
status: planned
dependencies: [M002-C007, M002-C004, M002-C005]
---

# M002-C003: Measure sustained CPU, GPU, and NPU-visible memory bandwidth

## Dependencies

- [M002-C007](007-machine-fingerprint-artifact.md)
- [M002-C004](004-hip-deterministic-smoke.md)
- [M002-C005](005-xrt-deterministic-smoke.md)

## Required Context

- docs/ROADMAP.md
- docs/PERFORMANCE.md
- docs/BENCHMARKS.md
- docs/MEMORY_MANAGEMENT.md
- docs/GPU_BACKEND.md
- docs/NPU_BACKEND.md

## Goal

Add a reproducible bandwidth benchmark that measures sustained CPU memory, HIP device-visible memory, and XRT/NPU-visible buffer transfer/access paths and emits retained, fingerprinted structured results.

## Non-Goals

- Claiming theoretical LPDDR bandwidth
- GPU/NPU simultaneous-load or dma-buf interoperability proof
- Model kernel benchmarking
- Choosing heterogeneous routes
- Promoting a result lacking a machine fingerprint or raw repetitions

## Expected Paths

- `src/core/diagnostics/bandwidth.cpp`
- `src/core/diagnostics/bandwidth.h`
- `src/core/hip/bandwidth.hip`
- `src/core/xdna2/bandwidth.cpp`
- `tests/diagnostics/bandwidth_test.cpp`
- `tests/fixtures/diagnostics/bandwidth_schema.json`
- `benchmarks/hardware/README.md`
- `docs/CLI.md`
- `CMakeLists.txt`

## Definition of Done

- [ ] CPU reports copy/read/write bandwidth with explicit working set, threads, affinity policy, warmups, repetitions, bytes, and elapsed time
- [ ] HIP reports host-to-device, device-to-host, device-visible copy/read/write paths as applicable, with synchronization included and exact allocation type stated
- [ ] XRT/NPU-visible reporting distinguishes host BO map/sync/DMA/device-access paths and explicitly marks unsupported paths
- [ ] Working sets exceed relevant caches and measurements run long enough to report sustained median, p95, raw repetitions, temperature boundaries, and clock boundaries
- [ ] Each JSON result embeds or references the exact M002-C007 fingerprint ID and records command, revision, backend, route, warmup, repetitions, and checksum
- [ ] Correctness sentinels are verified before any bandwidth result counts
- [ ] A sanitized retained baseline summary and checksums are committed; raw host-specific artifacts are stored outside Git according to docs/TESTING.md
- [ ] One focused PR adds only bandwidth measurement, validation, schema, and baseline documentation

## Development Loop

```sh
nix develop -c cmake --preset test
nix develop -c cmake --build --preset test
git add -A
nix build
nix flake check
./result/bin/strix-server diagnose --benchmark bandwidth --backends cpu,hip,xrt --warmup 3 --repetitions 10 --duration-ms 2000 --json --output /tmp/strix-bandwidth.json
./result/bin/strix-server diagnose --validate-artifact /tmp/strix-bandwidth.json
nix develop -c ctest --preset test --output-on-failure -R 'bandwidth_schema|bandwidth_correctness'
# Run with normal production power policy and no unrelated load; do not report a pass if HIP or XRT was requested but skipped.
```

## Output Artifacts

- /tmp/strix-bandwidth.json (full raw benchmark artifact retained externally by content hash)
- benchmarks/hardware/README.md (sanitized baseline summary with artifact checksum and fingerprint ID)
- tests/fixtures/diagnostics/bandwidth_schema.json

## Stop Conditions

- M002-C007 fingerprint is missing or invalid
- A requested backend is skipped, silently falls back to CPU, or fails sentinel validation
- Thermal or clock drift invalidates sustained comparison
- The selected XRT operation does not represent an NPU-visible allocation/path and cannot be labeled precisely
- Results omit raw repetitions or include only the best run

## ROADMAP Traceability

- docs/ROADMAP.md Milestone 1 task 3
- docs/ROADMAP.md Milestone 1 exit criterion 3
- docs/PERFORMANCE.md#measurement-rules
- docs/TESTING.md#baseline-and-artifact-management

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
