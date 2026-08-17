---
id: M005-C021
title: "Implement transactional one-token decode and the greedy generation loop"
milestone: M005
status: planned
dependencies: [M005-C010, M005-C013, M005-C019, M005-C020]
---

# M005-C021: Implement transactional one-token decode and the greedy generation loop

## Dependencies

- [M005-C010](010-contiguous-kv-attention.md)
- [M005-C013](013-eager-full-model-executor.md)
- [M005-C019](019-qwen-deltanet-state-transactions.md)
- [M005-C020](020-deterministic-greedy-argmax.md)

## Required Context

- `docs/ROADMAP.md`, Milestone 4 eager GPU-only slice.
- `docs/TOKENIZATION.md`, greedy and finish-reason contracts.
- `docs/TESTING.md`, state commit, rollback, and cancellation injection.
- Prefill executor, KV state, DeltaNet state transactions, and greedy selector.

## Goal

Implement one eager GPU-only decode step and a bounded autoregressive loop that
commits exactly one token/state generation after success, stops deterministically
on EOS or output limit, and rolls back cleanly on failure or cancellation.

## Non-Goals

- CLI parsing or terminal output.
- Stochastic sampling, chat, HTTP, batching, paged KV, graphs, speculation, or NPU.
- Performance tuning beyond enforcing no allocations in the timed step.

## Expected Paths

- Add: `models/qwen35_08b/runtime/decode_executor.*`.
- Add: `src/core/generation/greedy_generation.*`.
- Add: `tests/integration/qwen35_08b_generation_loop_test.*`.
- Produce: `artifacts/m005/generation-loop-trace.json`.

## Definition of Done

- [ ] One decode step executes all 24 blocks, final normalization, LM head, and greedy argmax from the committed prior state.
- [ ] A successful step commits exactly one token, KV position, DeltaNet state generation, and output cursor atomically.
- [ ] Injected kernel error, timeout, cancellation, and non-finite logits leave token history and canonical state byte-identical to the pre-step snapshot.
- [ ] EOS and maximum-output-token termination have explicit deterministic precedence and finish reasons.
- [ ] Repeated execution from identical prompt/state produces identical token IDs and state hashes.
- [ ] Allocation instrumentation reports zero general/device allocations inside the timed decode loop.

## Development Loop

```sh
mkdir -p artifacts/m005
nix develop -c cmake --preset hip-test
nix develop -c cmake --build --preset hip-test
nix develop -c ctest --preset hip-test --output-on-failure -R 'qwen35_generation_loop'
```

## Hardware Validation

```sh
STRIX_ARTIFACT_DIR=artifacts/m005 STRIX_REQUIRE_GFX1151=1 nix develop -c ctest --preset hip-test --output-on-failure -R 'qwen35_generation_loop_(determinism|rollback|no_alloc)_hardware'
```

## Output Artifacts

- `artifacts/m005/generation-loop-trace.json`.
- Token/state-hash determinism report.
- Failure-injection rollback and no-allocation reports.

## Stop Conditions

- Stop at the first token or state-hash divergence.
- Stop if any failed step mutates canonical state.
- Stop if timed decode allocates, selects a non-GPU route, or requires graph replay.
- Stop before adding CLI or transport behavior.

## ROADMAP Traceability

- Milestone 4 tasks 2–5 and 10 for eager greedy direct execution; task 9 sampling/chat remains deferred.
- Exit: greedy token history is deterministic.
- Exit: no general allocation occurs in timed decode.
- Exit: cancellation reclaims provisional state.

## Agent Handoff

When complete, report files changed, exact commands/results, trace hashes,
failure-injection coverage, hardware identity, and residual risks without
expanding scope.
