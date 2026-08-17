---
id: M005-C012
title: "Implement final normalization and LM-head logits"
milestone: M005
status: planned
dependencies: [M003-C008, M005-C006, M005-C007]
---

# M005-C012: Implement final normalization and LM-head logits

## Dependencies

- [M003-C008](../003-test-oracles-model-contracts/008-high-precision-cpu-operator-oracles.md)
- [M005-C006](006-qwen-elementwise-hip-ops.md)
- [M005-C007](007-qwen-projection-hip-ops.md)

## Required Context

- The GPU model must produce numerically validated full-vocabulary logits before token selection. M005-C020 separately owns deterministic device-side greedy argmax and the bounded host result.

## Goal

Compute final normalized hidden state and full-vocabulary logits from the tied or untied LM head defined by the checked model descriptor, and validate them against the pinned oracle.

## Non-Goals

- Temperature
- Top-k/top-p
- RNG
- Greedy argmax or token commit; M005-C020 owns selection.
- Logprobs API.
- Structured output.
- MTP.

## Expected Paths

- read: `docs/TOKENIZATION.md#Sampling-Contract`
- read: `docs/GPU_BACKEND.md#Sampling`
- read: `docs/TESTING.md#Full-Logit-Testing`
- write: `models/qwen35_08b/lm_head.*`
- write: `tests/models/qwen35_08b/lm_head_test.*`

## Definition of Done

- [ ] Final normalization and LM-head logits match declared full-logit oracle thresholds for pinned fixtures.
- [ ] Tied-embedding behavior for 0.8B and untied descriptor behavior for 27B are explicit and validated without executing 27B.
- [ ] Non-finite hidden values or logits fail with operation/index diagnostics.
- [ ] The kernel writes into preallocated device storage and performs no token selection, host transfer, or request-state commit.

## Development Loop

```sh
mkdir -p artifacts/m005
nix develop -c cmake --preset hip-test
nix develop -c cmake --build --preset hip-test
git add -A
nix build
nix develop -c ctest --preset hip-test --output-on-failure -R 'qwen35_lm_head_logits'
```

## Hardware Validation

```sh
STRIX_ARTIFACT_DIR=artifacts/m005 STRIX_REQUIRE_GFX1151=1 nix develop -c ctest --preset hip-test --output-on-failure -R 'qwen35_lm_head_logits_hardware'
```

## Output Artifacts

- Full-logit comparison artifact
- Full-logit finite-value and numerical-comparison report.

## Stop Conditions

- Stop on a non-finite logit or a full-logit threshold failure.
- Stop before adding argmax, host transfer, or sampling-state behavior.

## ROADMAP Traceability

- tasks: M4.2 LM head
- exitCriteria: Full logits pass thresholds
- docs: docs/TOKENIZATION.md#Sampling-Pipeline
- docs: docs/TESTING.md#Full-Logit-Testing

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
