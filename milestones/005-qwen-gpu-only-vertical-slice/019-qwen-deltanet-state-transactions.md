---
id: M005-C019
title: "Implement request-owned DeltaNet state and transactional prefill/decode"
milestone: M005
status: planned
dependencies: [M005-C005, M005-C008]
---

# M005-C019: Implement request-owned DeltaNet state and transactional prefill/decode

## Dependencies

- [M005-C005](005-request-state-and-gpu-arenas.md)
- [M005-C008](008-qwen-gated-deltanet-state.md)

## Required Context

- `docs/TESTING.md`, layer/state boundaries and cancellation injection.
- `docs/SCHEDULER.md`, commit and rollback ownership rules applicable to direct execution.
- Qwen DeltaNet CPU state fixtures from Milestone 2.
- Stateless numerical primitives produced by M005-C008.

## Goal

Own convolution and recurrent state per request, implement prefill and one-token
state transitions, and publish state only after successful eager HIP execution.

## Non-Goals

- New DeltaNet math or kernel tuning.
- Generic SSM support.
- Continuous batching, snapshots, speculation, or NPU execution.
- Full-model orchestration.

## Expected Paths

- Add: `models/qwen35_08b/state/deltanet_state.*`.
- Add: `models/qwen35_08b/runtime/deltanet_execution.*`.
- Add: `tests/models/qwen35_08b/deltanet_state_transaction_test.*`.
- Produce: `artifacts/m005/deltanet-state-comparison.json`.

## Definition of Done

- [ ] State sizes and offsets derive from the checked model descriptor and are validated for all 18 0.8B DeltaNet layers.
- [ ] Prefill and token-serial execution match pinned output, convolution-state, and recurrent-state fixtures at every selected boundary.
- [ ] Position, initialized length, generation, and owning request are explicit and checked before dispatch.
- [ ] Injected launch, timeout, cancellation, and comparison failures leave canonical state byte-identical to its pre-step value.
- [ ] Successful commit publishes exactly one new state generation and performs no allocation inside timed execution.
- [ ] Descriptor-only Qwen3.8-27B state arithmetic validates without allocating weights or state.

## Development Loop

```sh
mkdir -p artifacts/m005
nix develop -c cmake --preset hip-test
nix develop -c cmake --build --preset hip-test
nix develop -c ctest --preset hip-test --output-on-failure -R 'qwen35_deltanet_state_transaction'
```

## Hardware Validation

```sh
STRIX_ARTIFACT_DIR=artifacts/m005 STRIX_REQUIRE_GFX1151=1 nix develop -c ctest --preset hip-test --output-on-failure -R 'qwen35_deltanet_(prefill|decode|rollback)_hardware'
```

## Output Artifacts

- `artifacts/m005/deltanet-state-comparison.json`.
- Failure-injection rollback report.
- State-size and allocation-accounting report.

## Stop Conditions

- Stop at the first state mismatch even if the final layer output remains within tolerance.
- Stop if an error path mutates canonical state or requires reconstructing it from output.
- Stop if state ownership cannot be represented without introducing scheduler/batching scope.

## ROADMAP Traceability

- Milestone 4 task 2: linear-attention/Gated DeltaNet execution required by the Qwen model contract.
- Milestone 4 task 5: request-owned model state.
- Exit: layer/state outputs pass thresholds.
- Exit: cancellation reclaims provisional state.

## Agent Handoff

When complete, report files changed, exact commands/results, state comparison
artifact hashes, injected-failure coverage, skipped hardware checks, and
residual risks without expanding scope.
