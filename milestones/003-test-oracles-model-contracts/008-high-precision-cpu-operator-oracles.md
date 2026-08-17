---
id: M003-C008
title: "Add small high-precision CPU operator oracles"
milestone: M003
status: planned
dependencies: [M003-C001, M003-C004]
---

# M003-C008: Add small high-precision CPU operator oracles

## Dependencies

- [M003-C001](001-checked-tensor-descriptor.md)
- [M003-C004](004-qwen35-08b-descriptor.md)

## Required Context

- Accelerator work in Milestone 3 needs an oracle independent of future HIP/AIE kernels. These are intentionally small analytic fixtures, not a performant CPU model runtime.

## Goal

Implement deterministic CPU reference operators for Qwen-required matvec/matmul, RMSNorm, RoPE, softmax/attention primitives, gated activation, residual, dynamic A8 quantization, and wide-integer group accumulation with explicit storage/input/accumulation/output dtype contracts.

## Non-Goals

- Do not build a full CPU inference engine.
- Do not optimize with backend-specific SIMD or reuse future device kernels.
- Do not add GPU/NPU code.

## Expected Paths

- `models/qwen35_08b/cpu/oracles.hpp`
- `models/qwen35_08b/cpu/oracles.cpp`
- `tests/models/qwen_cpu_oracles_test.cpp`
- `tests/fixtures/cpu/qwen-operators.json`
- `CMakeLists.txt`

## Definition of Done

- [ ] Each operator states source/input, intermediate accumulation, reduction order, and output dtype; tests prove source BF16 is not mislabeled FP32.
- [ ] Hand-checkable fixtures include normal, zero, tail, saturation, non-finite rejection, odd dimensions, RoPE boundaries, and exact INT32/wider accumulation cases.
- [ ] Floating references use FP64 where analytic and compare declared BF16/FP32 outputs under justified exactness/tolerances.
- [ ] The implementation has no dependency on HIP, XRT, torch, or accelerator numerical source.

## Development Loop

```sh
nix develop -c bash -lc 'cmake -S . -B build/m003-c008 -DBUILD_TESTING=ON && cmake --build build/m003-c008 --target qwen_cpu_oracles_test && ctest --test-dir build/m003-c008 --output-on-failure -R ^qwen_cpu_oracles_test$'
git add -A
nix build -L
```

## Output Artifacts

- Small analytic operator fixtures
- CPU oracle numerical-contract report

## Stop Conditions

- Stop if a future accelerator implementation becomes its own oracle.
- Stop if source storage dtype and accumulation dtype are conflated.
- Stop before full-model CPU execution or performance tuning.

## ROADMAP Traceability

- M2 task 6
- M2 exits 3 and 5
- TESTING.md Oracle Hierarchy
- TESTING.md T1-T2

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
