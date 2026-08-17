---
id: M005-C011
title: "Implement the complete Qwen gated MLP block"
milestone: M005
status: planned
dependencies: [M003-C008, M005-C006, M005-C007]
---

# M005-C011: Implement the complete Qwen gated MLP block

## Dependencies

- [M003-C008](../003-test-oracles-model-contracts/008-high-precision-cpu-operator-oracles.md)
- [M005-C006](006-qwen-elementwise-hip-ops.md)
- [M005-C007](007-qwen-projection-hip-ops.md)

## Required Context

- Projection kernels alone do not define activation order, gating, numerical accumulation, residual boundaries, or tensor binding for the model's FFN.

## Goal

Compose RMSNorm, gate/up projections, pinned activation and elementwise gate, down projection, and residual update into an eager model-private MLP block.

## Non-Goals

- Alternative activations
- Fusion justified only by microbenchmark
- SHQ4 rollout
- MoE
- 27B execution

## Expected Paths

- read: `tools/strix/model.py`
- read: `docs/GPU_BACKEND.md`
- read: `docs/TESTING.md`
- write: `models/qwen35_08b/mlp.*`
- write: `models/qwen35_08b/gpu/gfx1151/mlp*.hip`
- write: `models/qwen35_08b/cpu/oracles/mlp.*`
- write: `tests/models/qwen35/mlp_test.*`

## Definition of Done

- [ ] Gate, up, activated product, down, and residual boundaries match declared oracle thresholds for representative layers and tails.
- [ ] Tensor orientation and encoding come only from the compiled descriptor.
- [ ] Scratch is bounded and reused with no timed allocation.
- [ ] All 24 0.8B layers bind their MLP tensors.

## Development Loop

```sh
mkdir -p artifacts/m005
nix develop -c cmake --preset hip-test
nix develop -c cmake --build --preset hip-test
git add -A
nix build
nix develop -c ctest --preset hip-test --output-on-failure -R 'qwen35_mlp_(oracle|binding)'
```

## Hardware Validation

```sh
STRIX_ARTIFACT_DIR=artifacts/m005 STRIX_REQUIRE_GFX1151=1 nix develop -c ctest --preset hip-test --output-on-failure -R 'm005_c011_hardware'
```

## Output Artifacts

- MLP boundary comparison artifact
- MLP scratch high-water report

## Stop Conditions

- Stop on activation-order or tensor-orientation ambiguity.
- Stop if fusion removes required debug boundaries before full-model acceptance.

## ROADMAP Traceability

- tasks: M4.2 gate, up, down projections and residual paths
- exitCriteria: Layer outputs pass oracle thresholds
- exitCriteria: No general allocation in timed decode
- docs: docs/TESTING.md#T3:-Layer-and-state-boundary-tests

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
