---
id: M005-C008
title: "Implement Qwen Gated DeltaNet numerical primitives"
milestone: M005
status: planned
dependencies: [M003-C008, M005-C005, M005-C006, M005-C007]
---

# M005-C008: Implement Qwen Gated DeltaNet numerical primitives

## Dependencies

- [M003-C008](../003-test-oracles-model-contracts/008-high-precision-cpu-operator-oracles.md)
- [M005-C005](005-request-state-and-gpu-arenas.md)
- [M005-C006](006-qwen-elementwise-hip-ops.md)
- [M005-C007](007-qwen-projection-hip-ops.md)

## Required Context

- Eighteen of the 24 Qwen3.5-0.8B blocks are Gated DeltaNet blocks. This card owns only the model-private numerical primitives and one-step outputs; M005-C019 owns persistent recurrent state and transactional prefill/decode publication.

## Goal

Implement and validate the model-private HIP convolution, gates, normalization, and one-step DeltaNet recurrence primitives against the Milestone 2 CPU oracle and boundary fixtures.

## Non-Goals

- A generic SSM framework
- MTP
- NPU execution
- Persistent request-state ownership or rollback; M005-C019 owns it.
- Full prefill/decode orchestration.
- Continuous batching.
- 27B execution.

## Expected Paths

- read: `benchmarks/qwen3.5-0.8b/README.md`
- read: `tools/strix/model.py`
- read: `docs/TESTING.md`
- write: `models/qwen35_08b/gpu/gfx1151/deltanet*.hip`
- read: `models/qwen35_08b/cpu/oracles/deltanet.*`
- write: `tests/models/qwen35/deltanet_primitives_test.*`

## Definition of Done

- [ ] Convolution, gate, normalization, and one-step recurrence outputs match the pinned CPU oracle and committed boundary fixtures.
- [ ] Minimum, representative, odd-tail, zero-state, and non-finite cases are covered with operation-specific tolerances.
- [ ] All 18 expected 0.8B DeltaNet layers bind their exact tensor shapes without hidden global constants.
- [ ] The primitive API accepts explicit input/output/state views and performs no allocation or logical state commit.
- [ ] Descriptor-only 27B shape validation reuses interfaces without dispatch.

## Development Loop

```sh
mkdir -p artifacts/m005
nix develop -c cmake --preset hip-test
nix develop -c cmake --build --preset hip-test
git add -A
nix build
nix develop -c ctest --preset hip-test --output-on-failure -R 'qwen35_deltanet_primitives'
```

## Hardware Validation

```sh
STRIX_ARTIFACT_DIR=artifacts/m005 STRIX_REQUIRE_GFX1151=1 nix develop -c ctest --preset hip-test --output-on-failure -R 'qwen35_deltanet_primitives_hardware'
```

## Output Artifacts

- DeltaNet primitive boundary comparison artifact.
- Operation-specific numerical error report.

## Stop Conditions

- Stop at the first divergent primitive boundary; do not hide a state/update mismatch behind final-output tolerance.
- Stop if the reference semantics for the pinned model revision are ambiguous.
- Stop before adding request ownership, rollback, or complete prefill/decode sequencing.

## ROADMAP Traceability

- tasks: M4.2 model-private Qwen GPU operations
- exitCriteria: Layer outputs pass thresholds
- docs: docs/PROJECT_STATUS.md#Models
- docs: docs/TESTING.md#T3:-Layer-and-state-boundary-tests

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
