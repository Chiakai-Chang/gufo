---
id: M005-C005
title: "Create request-owned state and preallocated GPU arenas"
milestone: M005
status: planned
dependencies: [M005-C001, M005-C002, M005-C003]
---

# M005-C005: Create request-owned state and preallocated GPU arenas

## Dependencies

- [M005-C001](001-gpu-only-strix-server-target.md)
- [M005-C002](002-qwen-compiled-descriptors.md)
- [M005-C003](003-transactional-text-artifact-loader.md)

## Required Context

- MVP is one direct request, but model, token, sampling, recurrent, contiguous-KV, scratch, and cancellation state must have explicit ownership and rollback. Timed decode cannot allocate.

## Goal

Implement one-request ownership records, generation-tagged handles, aligned immutable/persistent/KV/scratch arenas, reservation-before-transition, and idempotent cancellation rollback.

## Non-Goals

- Multi-request scheduler
- Paged KV
- Prefix cache
- Speculative provisional pages
- NPU-visible allocations

## Expected Paths

- read: `docs/MEMORY_MANAGEMENT.md`
- read: `docs/KV_CACHE.md`
- read: `docs/SCHEDULER.md`
- write: `src/core/memory/request_arena.*`
- write: `src/core/request/request_state.*`
- write: `tests/memory/request_arena_test.*`
- write: `tests/request_state_test.*`

## Definition of Done

- [ ] Every mutable model, recurrent, KV, decode, and greedy state is owned by exactly one request.
- [ ] All worst-case MVP arenas are reserved before prefill; timed token transitions perform zero general allocations.
- [ ] Cancellation and every injected backend failure return memory accounting to the pre-request baseline.
- [ ] Stale generation-tagged handles are rejected after slot reuse.

## Development Loop

```sh
mkdir -p artifacts/m005
nix develop -c cmake --preset hip-test
nix develop -c cmake --build --preset hip-test
git add -A
nix build
nix develop -c ctest --preset hip-test --output-on-failure -R 'request_(ownership|rollback)|arena_(bounds|generation)'
```

## Hardware Validation

```sh
STRIX_ARTIFACT_DIR=artifacts/m005 STRIX_REQUIRE_GFX1151=1 nix develop -c ctest --preset hip-test --output-on-failure -R 'm005_c005_(arena|no_alloc|reset)_stress'
```

## Output Artifacts

- Allocation-class budget JSON
- Cancellation/fault rollback accounting report

## Stop Conditions

- Stop if any state is global, physical-row-owned, or shared implicitly between requests.
- Stop if a timed decode allocation or first-touch page fault is observed.

## ROADMAP Traceability

- tasks: M4.3 initial contiguous KV allocation prerequisite
- tasks: M4.5 request-owned model and sampling state
- exitCriteria: No general allocation in timed decode
- exitCriteria: Cancellation reclaims provisional state
- docs: docs/MEMORY_MANAGEMENT.md#KV-and-Request-Reservations
- docs: docs/KV_CACHE.md#Allocation

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
