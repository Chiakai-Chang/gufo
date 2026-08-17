---
id: M005-C018
title: "Run and retain the complete M4 gfx1151 acceptance record"
milestone: M005
status: planned
dependencies: [M002-C007, M005-C003, M005-C005, M005-C014, M005-C016, M005-C017, M005-C021, M005-C022]
---

# M005-C018: Run and retain the complete M4 gfx1151 acceptance record

## Dependencies

- [M002-C007](../002-hardware-diagnostics-baselines/007-machine-fingerprint-artifact.md)
- [M005-C003](003-transactional-text-artifact-loader.md)
- [M005-C005](005-request-state-and-gpu-arenas.md)
- [M005-C014](014-layer-logit-oracle-gates.md)
- [M005-C016](016-strix-server-prompt-cli.md)
- [M005-C017](017-exact-token-and-eval-drift.md)
- [M005-C021](021-one-token-decode-generation-loop.md)
- [M005-C022](022-capability-drift-trace-regrade.md)

## Required Context

- The milestone exits only with correctness-linked TTFT, inter-token latency, memory, determinism, cancellation, and terminal evidence on the supported machine. Performance cannot excuse correctness failure.

## Goal

Create one reproducible Nix-driven M4 acceptance command and retain the complete GPU-only 0.8B machine-fingerprinted report, including no-allocation instrumentation and thermal-aware single-request baselines.

## Non-Goals

- Performance target invention
- 27B runtime benchmark
- NPU comparison
- SHQ4 optimization
- HIP graphs
- Chat or sampling acceptance

## Expected Paths

- read: `docs/PERFORMANCE.md`
- read: `docs/TESTING.md`
- read: `docs/ROADMAP.md`
- read: `docs/PROJECT_STATUS.md`
- write: `tests/performance/m4_gpu_vertical_slice.*`
- write: `tests/integration/m4_acceptance.*`
- write: `tests/performance/m005_generation_baseline.*`
- write: `tests/integration/m005_acceptance.*`
- write: `tests/fixtures/m005/acceptance.schema.json`
- write: `tools/validate-m005-report.py`
- write: generated artifact schema only; no host-specific benchmark result committed

## Definition of Done

- [ ] A clean Nix build and all applicable T0-T4/T6 targeted gates pass on gfx1151 with route GPU_ONLY and eager dispatch.
- [ ] A terminal prompt from the checksummed 0.8B safetensors-derived artifact emits expected exact tokens repeatedly.
- [ ] Layer and full logits pass declared thresholds; any mismatch fails before timing.
- [ ] Allocation/page-fault instrumentation reports zero general allocations and zero first-touch model faults in timed decode.
- [ ] Cancellation at every commit boundary returns memory to baseline and exits 130 when invoked by SIGINT.
- [ ] Machine-fingerprinted artifacts retain TTFT, prefill time, per-token latency distribution, tokens/s, peak/current memory, clocks, thermal state, ROCm/compiler/driver revisions, warmup, repetitions, model/artifact/suite hashes.
- [ ] Dependency scan attests that the strix-server inference closure contains no NPU dependency.
- [ ] The milestone report explicitly records sampling, chat, HIP graphs, SHQ4, 27B execution, and NPU as not implemented/deferred rather than silently passing them.

## Development Loop

```sh
nix develop -c cmake --preset hip-test
nix develop -c cmake --build --preset hip-test
git add -A
nix build
STRIX_M005_ARTIFACT=artifacts/runtime/qwen35-08b \
  nix develop -c ctest --preset hip-test --output-on-failure -L m4-pr
mkdir -p artifacts/m005
nix develop -c env PYTHONPATH=tools python3 tools/validate-m005-report.py \
  --input artifacts/m005/acceptance.json \
  --schema tests/fixtures/m005/acceptance.schema.json
```

## Hardware Validation

```sh
mkdir -p artifacts/m005
STRIX_REQUIRE_GFX1151=1 STRIX_M005_ARTIFACT=artifacts/runtime/qwen35-08b \
  nix develop -c ctest --preset hip-test --output-on-failure -L m4-acceptance
nix develop -c bash -lc './result/bin/strix-server prompt --config tests/fixtures/cli/qwen08b.toml --model qwen35-08b --greedy --prompt-file tests/models/qwen35/prompts/smoke.txt --show-token-ids > artifacts/m005/terminal.txt'
```

## Output Artifacts

- Validated m4-gpu-vertical-slice-v1 acceptance manifest
- Machine fingerprint
- Exact terminal token record
- Layer/full-logit gate report
- TTFT/inter-token/memory baseline with raw repetitions
- No-allocation/page-fault trace
- Cancellation reclamation report
- No-NPU/no-CUDA dependency attestation

## Stop Conditions

- Stop immediately on any correctness, determinism, checksum, cancellation, or allocation gate failure; do not collect a promotable performance result.
- Stop if hardware is not gfx1151 or if driver/ROCm/compiler identity is missing.
- Stop if the inference route touches XRT/amdxdna or reports anything other than GPU_ONLY/eager.
- Stop if the baseline artifact lacks raw repetitions or a complete machine/model/suite fingerprint.

## ROADMAP Traceability

- tasks: M4.1-M4.8 integrated acceptance
- deferred: M4.9 sampling/chat receives no completion credit in this greedy acceptance gate
- tasks: M4.10 eager accepted; HIP graphs deferred
- exitCriteria: Safetensors-derived artifact produces text from terminal
- exitCriteria: Greedy token history deterministic
- exitCriteria: Layer outputs and full logits pass thresholds
- exitCriteria: No general allocation in timed decode
- exitCriteria: Cancellation reclaims provisional state
- exitCriteria: TTFT/inter-token/memory baselines retained
- docs: docs/ROADMAP.md#Milestone-4:-Qwen-GPU-Only-Vertical-Slice
- docs: docs/TESTING.md#T7:-End-to-end-performance
- docs: docs/PERFORMANCE.md

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
