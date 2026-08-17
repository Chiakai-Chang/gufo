---
id: M004-C010
title: "Run sustained GPU/NPU concurrency and close the non-blocking NPU evidence track"
milestone: M004
status: planned
dependencies: [M004-C005, M004-C006, M004-C007, M004-C008, M004-C012]
---

# M004-C010: Run sustained GPU/NPU concurrency and close the non-blocking NPU evidence track

**Track:** shared-allocation

## Dependencies

- [M004-C005](005-qwen-aie2p-xrt-program-lifecycle.md)
- [M004-C006](006-qwen-aie2p-w8a8-shq8-matrix.md)
- [M004-C007](007-qwen-aie2p-w4a8-shq4-matrix.md)
- [M004-C008](008-qwen-aie2p-bf16-shape-buckets.md)
- [M004-C012](012-xrt-hip-handoff-stress-and-tier.md)

## Required Context

- Concurrent ROCm+XDNA2 operation has plausible throughput benefit and a known firmware-timeout risk. This non-blocking track needs sustained evidence that protects GPU decode and disables unsafe or unhelpful NPU routes. M004-C011 independently owns the blocking GPU-format gate; SHQ remains candidate.

## Goal

Execute sustained independent and dependency-coupled GPU/NPU load matrices, then emit a machine-checkable NPU concurrency disposition without changing the independently closed GPU-format gate.

## Non-Goals

- Implementing a production heterogeneous scheduler
- Split-N or Split-K production routes
- Per-layer ping-pong
- Treating simultaneous activity as a performance win
- Any SHQ freeze or revision decision.
- Re-evaluating the GPU-format gate.
- Blocking Milestone 4 GPU-only work on NPU or interoperability.

## Expected Paths

- `tests/integration/test_shared_immutable_reads.cpp`
- `tests/integration/test_gpu_npu_concurrency.cpp`
- `tools/testing/npu_concurrency_stress.cpp`
- `tools/testing/m004_npu_report.py`
- `artifacts/m004/npu-concurrency-decision.schema.json`
- `CMakeLists.txt`
- `docs/generated/m004-npu-concurrency-summary.md`

## Definition of Done

- [ ] After producer completion, immutable simultaneous GPU/NPU reads are stressed and checked; all concurrent writes remain rejected.
- [ ] Concurrency matrix measures GPU-only, NPU-only, independent concurrent, dependency-coupled GPU-to-NPU, and dependency-coupled NPU-to-GPU modes where supported, under sustained thermal steady state.
- [ ] Report records throughput, TTFT/inter-token p50/p95/p99 where a representative protected-decode workload exists, device activity, bandwidth, power/thermal state, dispatch/handoff counts, firmware stalls, timeouts, resets, corruption, and recovery.
- [ ] Any candidate NPU/heterogeneous route must improve its declared end-to-end metric by at least 15 percent and regress protected GPU decode by less than 5 percent after synchronization, padding, repacking, duplicate memory, and dispatch costs; otherwise it is disabled.
- [ ] The report consumes the Tier A/B/C/D decision from M004-C012 and separately accepts or rejects each measured concurrency route. Tier D and rejection are valid completed outcomes.
- [ ] A firmware timeout, reset, corruption event, or unstable thermal result closes the route as disabled with retained evidence rather than keeping this card indefinitely open.
- [ ] Generated summary distinguishes microbenchmark evidence from route promotion, states that SHQ remains candidate, and does not publish model artifacts.

## Development Loop

```sh
git add -A
nix build
nix develop -c cmake -S . -B build-m004-c010 -DENGINE_ENABLE_HIP=ON -DENGINE_ENABLE_XRT=ON -DGPU_TARGETS=gfx1151 -DBUILD_TESTING=ON
nix develop -c cmake --build build-m004-c010 --target npu-concurrency-stress
nix develop -c ctest --test-dir build-m004-c010 --output-on-failure -R 'shared_immutable_reads|gpu_npu_concurrency|m004_exit_schema'
mkdir -p artifacts/m004 docs/generated
nix develop -c ./build-m004-c010/npu-concurrency-stress --scenario sustained-gpu-npu --modes gpu-only,npu-only,independent,gpu-to-npu,npu-to-gpu --protect-gpu-decode --thermal-steady-state --json artifacts/m004/c010-concurrency.json
nix develop -c env PYTHONPATH=tools python3 tools/testing/m004_npu_report.py --inputs artifacts/m004 --schema artifacts/m004/npu-concurrency-decision.schema.json --json artifacts/m004/npu-concurrency-decision.json --markdown docs/generated/m004-npu-concurrency-summary.md
```

## Hardware Validation

- Requires one supported Strix Halo host with gfx1151 and XDNA2 plus pinned software/firmware for positive concurrency evidence. If NPU or interop is unavailable or unstable, retain the supported-negative disabled decision; M004-C011 remains the independent GPU gate.

## Output Artifacts

- artifacts/m004/c010-concurrency.json
- artifacts/m004/npu-concurrency-decision.json
- docs/generated/m004-npu-concurrency-summary.md
- linked machine fingerprint and raw repetition data

## Stop Conditions

- Stop concurrent load immediately on corruption, firmware stall, timeout, reset, or unsafe thermal/power state; quarantine NPU and select GPU-only fallback.
- Disable any route that misses either the 15 percent end-to-end improvement or less-than-5-percent protected-decode regression gate.
- Do not make a SHQ stability decision in this card.
- Do not hold the GPU MVP or Milestone 4 GPU-only slice for NPU recovery, one-copy viability, or route promotion.
- Do not publish portable artifacts from a retain-candidate/defer disposition.

## ROADMAP Traceability

- docs/ROADMAP.md#milestone-3-early-gpu-and-npu-format-feasibility — shared-allocation item 6 only and non-blocking NPU exit
- docs/PROJECT_STATUS.md#promotion-gates
- docs/PROJECT_STATUS.md#shq-format-stability
- docs/NPU_RESEARCH.md#promotion-policy
- docs/HETEROGENEOUS_EXECUTION.md#promotion-requirements
- docs/TESTING.md#t7-end-to-end-performance

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
