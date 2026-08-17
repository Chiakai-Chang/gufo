---
id: M004-C002
title: "Implement and characterize Qwen-private SHQ4-T16 decode GEMV"
milestone: M004
status: planned
dependencies: [M001-C008, M002-C007, M003-C008, M003-C011, M003-C013]
---

# M004-C002: Implement and characterize Qwen-private SHQ4-T16 decode GEMV

**Track:** gpu

## Dependencies

- [M001-C008](../001-repository-toolchain/008-single-pr-test-entrypoint.md)
- [M002-C007](../002-hardware-diagnostics-baselines/007-machine-fingerprint-artifact.md)
- [M003-C008](../003-test-oracles-model-contracts/008-high-precision-cpu-operator-oracles.md)
- [M003-C011](../003-test-oracles-model-contracts/011-candidate-shq-byte-vectors.md)
- [M003-C013](../003-test-oracles-model-contracts/013-milestone2-contract-gate.md)

## Required Context

- SHQ4 is the bandwidth-critical decode representation and requires fused unpack, zero correction, BF16 scale handling, and accumulation. Its arithmetic must be validated independently rather than using the new HIP path as its own oracle.

## Goal

Implement one Qwen-owned wave32 batch-1 SHQ4-T16 GEMV for U4Z G64 candidate planes and characterize correctness, ISA, occupancy, bandwidth, alignment, and tails on gfx1151.

## Non-Goals

- G32 or symmetric S4 promotion unless already required by the Milestone 2 contract
- SHQ6 production path
- Small-row/prefill kernels
- Cross-model source reuse
- Format freeze

## Expected Paths

- `CMakeLists.txt`
- `models/qwen35_08b/gpu/gfx1151/shq4_decode.hip`
- `models/qwen35_08b/gpu/gfx1151/shq4_decode.hpp`
- `tests/kernels/qwen35_08b/test_shq4_decode.cpp`
- `tests/fixtures/formats/shq4_t16/`
- `tools/testing/qwen_gfx1151_shq4_decode_gemv_bench.cpp`

## Definition of Done

- [ ] Kernel fuses nibble unpack, zero correction, BF16 scale conversion, and dot product while consuming canonical candidate planes directly.
- [ ] Exhaustive nibble/zero behavior plus minimum, typical, odd, padded, misaligned, and tail shapes match the independent wide CPU oracle under the declared contract.
- [ ] No allocation or repack occurs in timed dispatch; scratch is bounded and caller-owned.
- [ ] Retained JSON records gfx1151 ISA hash, resources, occupancy, effective bytes and bandwidth, launch latency, alignment/tail behavior, clocks, temperature, warmup, and raw repetitions.
- [ ] Kernel source and tuning remain under the Qwen model-private gfx1151 tree.

## Development Loop

```sh
mkdir -p artifacts/m004
git add -A
nix build
nix develop -c cmake -S . -B build-m004-c002 -DENGINE_ENABLE_HIP=ON -DENGINE_ENABLE_XRT=OFF -DGPU_TARGETS=gfx1151 -DBUILD_TESTING=ON
nix develop -c cmake --build build-m004-c002
STRIX_ARTIFACT_DIR=artifacts/m004 STRIX_REQUIRE_GFX1151=1 nix develop -c ctest --preset hip-test --output-on-failure -R 'm004_c002_hardware'
nix develop -c ctest --test-dir build-m004-c002 --output-on-failure -R qwen35_shq4_decode
```

## Hardware Validation

- Correctness/performance completion requires gfx1151 on pinned ROCm. CPU fixture generation remains independently runnable. Unsupported architecture must fail with an actionable diagnostic.

## Output Artifacts

- artifacts/m004/c002-gfx1151.json
- gfx1151 ISA/resource report
- byte-exact fixture identity and CPU/HIP comparison matrix

## Stop Conditions

- Stop before performance measurement on any oracle mismatch.
- Stop and keep SHQ4 unpromoted if malformed/tail accesses are detected by sanitizer or guard regions.
- Stop any optimization that introduces timed allocation, hidden repack, or cross-model numerical coupling.
- Do not declare the candidate bytes stable or v1.

## ROADMAP Traceability

- docs/ROADMAP.md#milestone-3-early-gpu-and-npu-format-feasibility — GPU track item 1
- docs/GPU_BACKEND.md#single-token-decode
- docs/QUANTIZATION.md#candidate-shq4-t16-contract
- docs/TESTING.md#t2-device-kernel-tests

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
