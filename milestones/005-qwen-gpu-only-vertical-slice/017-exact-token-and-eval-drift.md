---
id: M005-C017
title: "Add pinned exact-token terminal generation fixtures"
milestone: M005
status: planned
dependencies: [M005-C014, M005-C016, M005-C021]
---

# M005-C017: Add pinned exact-token terminal generation fixtures

## Dependencies

- [M005-C014](014-layer-logit-oracle-gates.md)
- [M005-C016](016-strix-server-prompt-cli.md)
- [M005-C021](021-one-token-decode-generation-loop.md)

## Required Context

- M4 requires direct CLI/reference exact-token fixtures. This card owns only deterministic token parity and provenance; M005-C022 owns capability traces, extractor self-tests, and offline regrade.

## Goal

Pin a bounded representative prompt set and prove that the native direct terminal path produces exactly the same greedy token IDs, finish reasons, and usage counts as the pinned reference candidate.

## Non-Goals

- Leaderboard claims
- Changing suite answers
- Capability grading, extractor logic, trace schema, or offline regrade; M005-C022 owns them.
- Sampling-based release gates.
- Full-suite mandatory promotion.
- 27B evaluation.

## Expected Paths

- read: `docs/EVAL.md`
- read: `docs/TESTING.md`
- read: `tools/suites/teacher.json`
- read: `tests/quality/`
- write: `tests/models/qwen35/prompts/`
- write: `tests/models/qwen35/expected/exact_tokens/`
- write: `tests/integration/exact_token_test.*`

## Definition of Done

- [ ] Pinned reference and native strix-server prompt produce identical greedy token IDs for the committed 0.8B fixture set.
- [ ] Repeated native runs produce identical token histories and finish reasons.
- [ ] Fixtures record source revision, candidate artifact ID, tokenizer/template hashes, prompt bytes, expected token IDs, finish reason, usage counts, and exporter command.
- [ ] Plain text, code, arithmetic, Unicode, repeated-token, and context-boundary prompts are represented.
- [ ] Five repeated native runs per fixture produce byte-identical token histories and finish reasons.
- [ ] Large generation traces remain under `artifacts/`; only bounded fixtures and provenance are committed.

## Development Loop

```sh
mkdir -p artifacts/m005
nix develop -c cmake --preset hip-test
nix develop -c cmake --build --preset hip-test
git add -A
nix build
nix develop -c ctest --preset hip-test --output-on-failure -R 'qwen35_exact_tokens'
```

## Hardware Validation

```sh
STRIX_ARTIFACT_DIR=artifacts/m005 STRIX_REQUIRE_GFX1151=1 nix develop -c ctest --preset hip-test --output-on-failure -R 'qwen35_exact_tokens_hardware'
```

## Output Artifacts

- Committed small exact-token fixtures with provenance
- Native/reference exact-token comparison report.
- Repetition determinism report.

## Stop Conditions

- Stop if reference and native prompts differ in tokenizer/template/model identity.
- Stop on any exact token or count drift even when decoded text or verdict is unchanged.
- Stop before adding capability grading or offline regrade behavior.

## ROADMAP Traceability

- tasks: M4.8 exact-token fixtures
- exitCriteria: Greedy token history is deterministic
- exitCriteria: Artifact produces terminal text
- docs: docs/TESTING.md#T4:-Model-and-route-tests

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
