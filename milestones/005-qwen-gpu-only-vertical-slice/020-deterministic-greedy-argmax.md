---
id: M005-C020
title: "Implement deterministic device-side greedy argmax"
milestone: M005
status: planned
dependencies: [M005-C005, M005-C012]
---

# M005-C020: Implement deterministic device-side greedy argmax

## Dependencies

- [M005-C005](005-request-state-and-gpu-arenas.md)
- [M005-C012](012-lm-head-greedy-argmax.md)

## Required Context

- `docs/TOKENIZATION.md`, greedy tie-breaking and sampling-state contract.
- `docs/GPU_BACKEND.md`, bounded device-to-host result policy.
- Full-logit device buffer and non-finite policy from M005-C012.

## Goal

Select the maximum finite logit on gfx1151 deterministically, resolve exact ties
with the lowest token ID, and return only the selected token plus bounded
diagnostics for transactional request commit.

## Non-Goals

- LM-head computation.
- Temperature, top-k, top-p, penalties, RNG, or structured output.
- Full-vocabulary host transfer.
- Generation-loop orchestration.

## Expected Paths

- Add: `models/qwen35_08b/gpu/gfx1151/greedy_argmax.hip`.
- Add: `models/qwen35_08b/gpu/gfx1151/greedy_argmax.*`.
- Add: `src/core/sampling/greedy_result.*`.
- Add: `tests/models/qwen35_08b/greedy_argmax_test.*`.
- Produce: `artifacts/m005/greedy-argmax-report.json`.

## Definition of Done

- [ ] Exact ties across lanes, workgroups, and vocabulary tails always select the lowest token ID.
- [ ] Negative-only, odd vocabulary length, NaN, positive/negative infinity, and all-non-finite fixtures have explicit tested behavior.
- [ ] Repeated runs over identical logits return byte-identical selected-token records.
- [ ] Ordinary execution transfers only token ID, selected logit if enabled, and bounded status metadata to the host.
- [ ] Sampling/request state changes only after a caller explicitly commits the successful result.

## Development Loop

```sh
mkdir -p artifacts/m005
nix develop -c cmake --preset hip-test
nix develop -c cmake --build --preset hip-test
nix develop -c ctest --preset hip-test --output-on-failure -R 'qwen35_greedy_argmax'
```

## Hardware Validation

```sh
STRIX_ARTIFACT_DIR=artifacts/m005 STRIX_REQUIRE_GFX1151=1 nix develop -c ctest --preset hip-test --output-on-failure -R 'qwen35_greedy_argmax_hardware'
```

## Output Artifacts

- `artifacts/m005/greedy-argmax-report.json`.
- Tie, vocabulary-tail, and non-finite fixture results.
- Bounded-transfer assertion report.

## Stop Conditions

- Stop on any unstable tie result or architecture-dependent token ID.
- Stop if implementation requires a full-vocabulary host copy.
- Stop before adding stochastic sampling or generation-loop behavior.

## ROADMAP Traceability

- Milestone 4 task 4: deterministic greedy sampling.
- Milestone 4 task 5: request-owned sampling state.
- Exit: greedy token history is deterministic.

## Agent Handoff

When complete, report files changed, exact commands/results, artifact hashes,
hardware identity, skipped checks, and residual risks without expanding scope.
