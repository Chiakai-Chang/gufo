---
id: M003-C005
title: "Reuse the Qwen descriptor family for Qwen3.8-27B text-only production metadata"
milestone: M003
status: planned
dependencies: [M003-C004]
---

# M003-C005: Reuse the Qwen descriptor family for Qwen3.8-27B text-only production metadata

## Dependencies

- [M003-C004](004-qwen35-08b-descriptor.md)

## Required Context

- Qwen3.8-27B is the first production model and shares the Qwen3.5 operator/descriptor family, but its dimensions, tensor inventory, policies, and acceptance identity must remain explicit. The first artifact is text-only and retains MTP.

## Goal

Instantiate the shared Qwen descriptor schema for the pinned Qwen3.8-27B revision with explicit 27B dimensions/inventory, text-only capability, vision exclusion, MTP retention, and a distinct model/artifact acceptance identity.

## Non-Goals

- Do not copy/fork the complete Qwen descriptor implementation.
- Do not infer a Qwen3.8 recipe from Qwen3.5 mixed-precision results.
- Do not implement multimodal inputs, kernels, loading, or inference.

## Expected Paths

- `models/qwen38_27b/qwen38_27b_descriptor.cpp`
- `tests/models/qwen38_27b/qwen38_27b_descriptor_test.cpp`
- `tests/fixtures/models/qwen38-27b-config.json`
- `tests/fixtures/models/qwen38-27b-inventory.json`
- `docs/PROJECT_STATUS.md`

## Definition of Done

- [ ] The 27B descriptor is data/configuration over the shared Qwen family API rather than a duplicated registry or parser.
- [ ] All dimensions, layer patterns, state contracts, tensor shapes/orientations, tokenizer/template IDs, and MTP roles are explicit and tested.
- [ ] model.visual.* is excluded with a recorded reason; image/video capability negotiation is rejected; MTP remains in inventory.
- [ ] The descriptor has independent kernel-policy, tuning-record, source-revision, and acceptance IDs despite family reuse.

## Development Loop

```sh
nix develop -c bash -lc 'cmake -S . -B build/m003-c005 -DBUILD_TESTING=ON && cmake --build build/m003-c005 --target qwen38_27b_descriptor_test && ctest --test-dir build/m003-c005 --output-on-failure -R ^qwen38_27b_descriptor_test$'
nix develop -c python3 -m unittest -v tests.tools.test_qwen_descriptor_reuse
```

## Output Artifacts

- Pinned Qwen3.8-27B text descriptor fixture
- Vision exclusion and MTP retention inventory record

## Stop Conditions

- Stop if implementation reuse erases model-specific dimensions, tuning, or acceptance identity.
- Stop if any multimodal capability is advertised by the initial artifact.
- Stop if MTP is discarded from the source contract.

## ROADMAP Traceability

- M2 tasks 3-4
- ROADMAP Objective
- PROJECT_STATUS.md First Artifact and Capability
- ROADMAP Out of Scope

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
