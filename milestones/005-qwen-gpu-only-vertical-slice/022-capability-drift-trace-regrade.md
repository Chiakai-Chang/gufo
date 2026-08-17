---
id: M005-C022
title: "Implement the capability drift trace and offline regrade"
milestone: M005
status: planned
dependencies: [M003-C012, M005-C016, M005-C017]
---

# M005-C022: Implement the capability drift trace and offline regrade

## Dependencies

- [M003-C012](../003-test-oracles-model-contracts/012-capability-suite-provenance-fixtures.md)
- [M005-C016](016-strix-server-prompt-cli.md)
- [M005-C017](017-exact-token-and-eval-drift.md)

## Required Context

- `docs/EVAL.md`, deterministic drift gate, trace schema, grading, and regrade.
- `docs/TESTING.md`, capability and route identity requirements.
- Capability suite/provenance fixtures from M003-C012.
- Exact-token native generation path from M005-C017.

## Goal

Implement extractor self-tests, the bounded deterministic capability drift
trace, and model-free offline regrade as a separate evaluation adapter over the
already validated direct generation path.

## Non-Goals

- Changing suite prompts, answers, grading policy, or thresholds.
- Leaderboard claims or a full-suite mandatory release gate.
- Sampling, 27B evaluation, or a second inference path.
- Exact-token fixture generation owned by M005-C017.

## Expected Paths

- Add: `src/eval/trace.*` and `src/eval/regrade.*`.
- Add: `tests/quality/eval_extractor_test.*` and `eval_regrade_test.*`.
- Add: `tests/fixtures/eval/`.
- Produce: `artifacts/m005/eval-drift.json` and regrade report.

## Definition of Done

- [ ] Extractor self-tests run without a model and cover ambiguous, wrapped, malformed, and normalization cases from `docs/EVAL.md`.
- [ ] The four-case greedy drift run records prompt/case ID, suite hash, model/artifact/route fingerprint, token IDs/count, extracted answer, verdict, timing, and bounded failure detail.
- [ ] Offline regrade reads only a prior trace plus grading fixtures, never loads model weights, and reports every changed case with the grading revision.
- [ ] Regrading an unchanged trace is deterministic and byte-stable after normalizing observation timestamps.
- [ ] Generated traces remain under `artifacts/`; committed files are bounded schema and extractor fixtures only.

## Development Loop

```sh
mkdir -p artifacts/m005
nix develop -c cmake --preset test
nix develop -c cmake --build --preset test
nix develop -c ctest --preset test --output-on-failure -R 'eval_(extractor|regrade)'
```

## Hardware Validation

```sh
STRIX_ARTIFACT_DIR=artifacts/m005 STRIX_REQUIRE_GFX1151=1 nix develop -c ctest --preset hip-test --output-on-failure -R 'eval_drift_trace_hardware'
```

Then run the model-free regrade test under the CPU-only `test` preset against
the captured trace.

## Output Artifacts

- `artifacts/m005/eval-drift.json`.
- Offline regrade report.
- Committed trace schema and extractor fixtures.

## Stop Conditions

- Stop if extractor behavior conflicts with committed grading fixtures.
- Stop if offline regrade loads weights, initializes HIP, or invokes generation.
- Stop if implementing this adapter requires a second model execution path.

## ROADMAP Traceability

- Milestone 4 task 6: capability drift gate, trace format, and offline regrade.
- Exit support: deterministic greedy behavior and retained route identity.
- `docs/EVAL.md#deterministic-drift-gate`.
- `docs/EVAL.md#trace-and-offline-regrade`.

## Agent Handoff

When complete, report files changed, exact commands/results, trace and regrade
hashes, hardware identity for capture, and residual risks without expanding
scope.
