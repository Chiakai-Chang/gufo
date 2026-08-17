---
id: M003-C001
title: "Implement the common tensor descriptor and checked arithmetic"
milestone: M003
status: planned
dependencies: [M001-C008]
---

# M003-C001: Implement the common tensor descriptor and checked arithmetic

## Dependencies

- [M001-C008](../001-repository-toolchain/008-single-pr-test-entrypoint.md)

## Required Context

- ROADMAP Milestone 2 task 1 requires a shared descriptor before model adapters or native artifact parsing. Candidate SHQ metadata also needs logical/padded shapes and plane ranges to be validated without integer wraparound.

## Goal

Add a small C++20 value type for dtype, rank, logical and padded extents, alignment, byte strides/ranges, and checked element/byte arithmetic, with structured errors for overflow and invalid shapes.

## Non-Goals

- Do not parse safetensors or SHQ containers.
- Do not add device allocation or GPU/NPU tensor classes.
- Do not infer model-specific tensor orientation.

## Expected Paths

- `src/core/tensor_descriptor.hpp`
- `src/core/tensor_descriptor.cpp`
- `tests/core/tensor_descriptor_test.cpp`
- `CMakeLists.txt`

## Definition of Done

- [ ] Descriptor construction rejects negative/zero-invalid extents, unsupported dtype, impossible alignment, range overflow, and overlapping/out-of-bounds planes.
- [ ] Checked multiply, add, product, round-up, and byte-count operations report errors rather than wrapping size_t/uint64_t.
- [ ] Tests cover scalars, zero-length policy, rank boundaries, odd shapes, SHQ N/K padding examples, and values at integer limits.
- [ ] No model-specific names or backend headers enter the core type.

## Development Loop

```sh
nix develop -c bash -lc 'cmake -S . -B build/m003-c001 -DBUILD_TESTING=ON && cmake --build build/m003-c001 --target tensor_descriptor_test && ctest --test-dir build/m003-c001 --output-on-failure -R ^tensor_descriptor_test$'
git add -A
nix build -L
```

## Output Artifacts

- Native checked descriptor API
- T0 checked-arithmetic test executable

## Stop Conditions

- Stop if any descriptor operation can silently overflow or truncate.
- Stop if the design requires a model-specific or accelerator-specific dependency.
- Stop after descriptor validation; leave container I/O to M003-C002 and later SHQ parsing milestones.

## ROADMAP Traceability

- M2 task 1
- M2 exit 1
- TESTING.md T0
- QUANTIZATION.md Candidate SHQ4-T16 Contract

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
