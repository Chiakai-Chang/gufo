---
id: M005-C015
title: "Implement the single-request direct greedy generation adapter"
milestone: M005
status: planned
dependencies: [M005-C004, M005-C005, M005-C021]
---

# M005-C015: Implement the single-request direct greedy generation adapter

## Dependencies

- [M005-C004](004-native-qwen-tokenizer-template.md)
- [M005-C005](005-request-state-and-gpu-arenas.md)
- [M005-C021](021-one-token-decode-generation-loop.md)

## Required Context

- The CLI must adapt to the same internal GenerationRequest contract rather than contain a second inference path. M005 needs one request and no batching window.

## Goal

Create a direct in-process GenerationRequest flow from prompt bytes through formatting/tokenization, admission/reservation, eager GPU prefill/decode, incremental detokenization, stop handling, statistics, and rollback.

## Non-Goals

- HTTP
- Client mode
- Continuous batching
- Chat history
- Sampling
- Structured output

## Expected Paths

- read: `docs/CLI.md`
- read: `docs/TOKENIZATION.md`
- read: `docs/SCHEDULER.md`
- write: `src/core/generation/generation_request.*`
- write: `src/server/direct_runtime.*`
- write: `tests/integration/direct_generation_test.*`

## Definition of Done

- [ ] Exactly one prompt source becomes one request-owned token/state stream with no batching delay.
- [ ] EOS, max tokens, cancellation, backend failure, and detokenization failure produce stable finish/error semantics.
- [ ] Ctrl-C cancellation can interrupt prefill or decode and returns all request memory to baseline.
- [ ] Statistics include GPU_ONLY route, prompt tokens, output tokens, TTFT, inter-token latency, and memory high-water.

## Development Loop

```sh
mkdir -p artifacts/m005
nix develop -c cmake --preset hip-test
nix develop -c cmake --build --preset hip-test
git add -A
nix build
nix develop -c ctest --preset hip-test --output-on-failure -R 'direct_generation_(success|stops|failure|cancel)'
```

## Hardware Validation

```sh
STRIX_ARTIFACT_DIR=artifacts/m005 STRIX_REQUIRE_GFX1151=1 nix develop -c ctest --preset hip-test --output-on-failure -R 'm005_c015_(cancel|rollback|memory)_stress'
```

## Output Artifacts

- Direct GenerationRequest lifecycle trace
- Stop precedence and cancellation accounting report

## Stop Conditions

- Stop if the adapter duplicates tokenizer, model, or sampling logic.
- Stop if cancellation waits for the entire requested generation or leaks committed/provisional state.

## ROADMAP Traceability

- tasks: M4.5 request-owned state
- tasks: M4.7 direct greedy mode
- exitCriteria: Cancellation reclaims all provisional state
- exitCriteria: Terminal generation prerequisite
- docs: docs/CLI.md#Direct-mode
- docs: docs/TOKENIZATION.md#Stop-Conditions

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
