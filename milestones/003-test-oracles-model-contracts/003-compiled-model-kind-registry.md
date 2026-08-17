---
id: M003-C003
title: "Add the compiled ModelKind registry"
milestone: M003
status: planned
dependencies: [M001-C008, M003-C001]
---

# M003-C003: Add the compiled ModelKind registry

## Dependencies

- [M001-C008](../001-repository-toolchain/008-single-pr-test-entrypoint.md)
- [M003-C001](001-checked-tensor-descriptor.md)

## Required Context

- The runtime supports curated compiled model kinds, not a general architecture loader. Registry resolution must be deterministic and must not derive architecture support from arbitrary artifact metadata.

## Goal

Implement a native registry with stable ModelKind identifiers, text capability declarations, descriptor lookup, and fail-closed resolution for the Qwen3.5 implementation family.

## Non-Goals

- Do not load weights or instantiate a model.
- Do not implement plugin/JIT/general architecture registration.
- Do not add kernels or backend selection.

## Expected Paths

- `src/core/model_kind.hpp`
- `src/core/model_registry.hpp`
- `src/core/model_registry.cpp`
- `tests/models/model_registry_test.cpp`
- `CMakeLists.txt`

## Definition of Done

- [ ] Stable IDs distinguish the Qwen3.5-0.8B iteration descriptor and Qwen3.8-27B production descriptor while linking both to one implementation-family ID.
- [ ] Registry lookup is deterministic, immutable after compilation, thread-safe, and rejects unknown/duplicate IDs.
- [ ] Capability flags declare text-only for the initial production descriptor and do not advertise image/video support.
- [ ] Tests prove that artifact-supplied strings cannot register executable behavior.

## Development Loop

```sh
nix develop -c bash -lc 'cmake -S . -B build/m003-c003 -DBUILD_TESTING=ON && cmake --build build/m003-c003 --target model_registry_test && ctest --test-dir build/m003-c003 --output-on-failure -R ^model_registry_test$'
git add -A
nix build -L
```

## Output Artifacts

- Compiled registry API
- Registry identity golden fixture

## Stop Conditions

- Stop if registry resolution requires arbitrary architecture code from an artifact.
- Stop if the two Qwen sizes collapse into one acceptance or tuning identity.
- Stop before weight loading or inference.

## ROADMAP Traceability

- M2 task 3
- ROADMAP Delivery Rules
- PROJECT_STATUS.md Models
- TESTING.md T0

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
