---
id: M004-C005
title: "Execute a minimal Qwen-private AIE2P program and measure XRT lifecycle"
milestone: M004
status: planned
dependencies: [M001-C008, M002-C005, M002-C007, M003-C011, M003-C013]
---

# M004-C005: Execute a minimal Qwen-private AIE2P program and measure XRT lifecycle

**Track:** npu

## Dependencies

- [M001-C008](../001-repository-toolchain/008-single-pr-test-entrypoint.md)
- [M002-C005](../002-hardware-diagnostics-baselines/005-xrt-deterministic-smoke.md)
- [M002-C007](../002-hardware-diagnostics-baselines/007-machine-fingerprint-artifact.md)
- [M003-C011](../003-test-oracles-model-contracts/011-candidate-shq-byte-vectors.md)
- [M003-C013](../003-test-oracles-model-contracts/013-milestone2-contract-gate.md)

## Required Context

- The current executable only enumerates XRT devices. Feasibility needs a reviewed ahead-of-time, model-private AIE2P artifact loaded through the supported XRT/amdxdna path, with lifecycle and assigned-resource costs measured before numerical kernels depend on it.

## Goal

Compile, embed, validate, load, repeatedly execute, and retire one deterministic Qwen-private AIE2P program through Strix-owned XRT abstractions while measuring context, program, submission, completion, and reconfiguration costs.

## Non-Goals

- Production NPU runtime
- Undocumented HSA packet dependency
- Raw driver ioctl spread
- GEMM performance claims
- Blocking GPU MVP when XDNA2 is unavailable

## Expected Paths

- `src/core/xdna2/device.cpp`
- `src/core/xdna2/context.cpp`
- `src/core/xdna2/program.cpp`
- `src/core/xdna2/completion.cpp`
- `models/qwen35_08b/npu/aie2p/smoke/`
- `tests/kernels/qwen35_08b/test_aie2p_smoke.cpp`
- `.devops/nix/package.nix`

## Definition of Done

- [ ] A reviewed source program is built ahead of time, embedded with architecture/compiler/program ABI/model-kind metadata and content hash, and unknown/substituted artifacts are rejected.
- [ ] C++ executes a deterministic device transform repeatedly through XRT with bounded completion waits and verifies output on the CPU.
- [ ] Report records device/driver/firmware/XRT/compiler IDs, requested and actually assigned columns, context creation, program load, command submission, completion, teardown, and repeated execution latency.
- [ ] Context and program creation remain outside the request-critical execution path; allocations return to baseline after repeated create/destroy cycles.
- [ ] No-XDNA2, ABI mismatch, timeout, or unsupported firmware yields a structured disabled-NPU result while GPU-only capability remains buildable and ready.

## Development Loop

```sh
mkdir -p artifacts/m004
git add -A
nix build
nix develop -c cmake -S . -B build-m004-c005 -DENGINE_ENABLE_HIP=OFF -DENGINE_ENABLE_XRT=ON -DBUILD_TESTING=ON
nix develop -c cmake --build build-m004-c005
STRIX_ARTIFACT_DIR=artifacts/m004 STRIX_REQUIRE_XDNA2=1 nix develop -c ctest --preset npu-test --output-on-failure -R 'm004_c005_hardware'
nix develop -c ctest --test-dir build-m004-c005 --output-on-failure -R qwen35_aie2p_smoke
```

## Hardware Validation

- Artifact build requires the pinned MLIR-AIE/IRON toolchain exposed by Nix. Device acceptance requires Linux x86_64 Strix Halo XDNA2/AIE2P with compatible amdxdna, XRT, and firmware. Absence closes as disabled/not-run, not as a GPU failure.

## Output Artifacts

- embedded AIE program content hash and metadata
- artifacts/m004/c005-xdna2.json
- disabled-capability record when unsupported

## Stop Conditions

- Stop after bounded timeout; quarantine the context and never count a reset as a pass.
- Reject an external or model-supplied executable artifact.
- Do not use undocumented HSA behavior as the production baseline.
- Do not delay GPU MVP due to NPU toolchain, driver, firmware, or hardware failure.

## ROADMAP Traceability

- docs/ROADMAP.md#milestone-3-early-gpu-and-npu-format-feasibility — NPU track item 1 and item 6
- docs/NPU_BACKEND.md#program-trust
- docs/NPU_BACKEND.md#program-build-pipeline
- docs/NPU_RESEARCH.md#configuration-reuse

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
