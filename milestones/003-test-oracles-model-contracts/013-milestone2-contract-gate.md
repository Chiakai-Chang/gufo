---
id: M003-C013
title: "Assemble the Milestone 2 oracle and contract exit gate"
milestone: M003
status: planned
dependencies: [M003-C002, M003-C003, M003-C004, M003-C005, M003-C006, M003-C007, M003-C008, M003-C009, M003-C010, M003-C011, M003-C012]
---

# M003-C013: Assemble the Milestone 2 oracle and contract exit gate

## Dependencies

- [M003-C002](002-safetensors-inspection-hardening.md)
- [M003-C003](003-compiled-model-kind-registry.md)
- [M003-C004](004-qwen35-08b-descriptor.md)
- [M003-C005](005-qwen38-27b-text-descriptor-reuse.md)
- [M003-C006](006-qwen-tokenizer-conformance.md)
- [M003-C007](007-qwen-chat-template-contract.md)
- [M003-C008](008-high-precision-cpu-operator-oracles.md)
- [M003-C009](009-versioned-teacher-logit-capture.md)
- [M003-C010](010-matched-token-logit-comparator.md)
- [M003-C011](011-candidate-shq-byte-vectors.md)
- [M003-C012](012-capability-suite-provenance-fixtures.md)

## Required Context

- Milestone 2 must leave one runnable, testable entry point that maps all five exits without claiming Milestone 3 accelerator evidence or SHQ v1 stability.

## Goal

Add a Nix-driven aggregate gate and machine-readable attestation that runs the complete offline/native Milestone 2 suite, validates fixture/provenance hashes, and explicitly reports each roadmap exit and numerical dtype contract.

## Non-Goals

- Do not add HIP/AIE numerical kernels, full model loading, generation, or server behavior.
- Do not download Qwen weights during the default gate.
- Do not freeze SHQ v1 or promote a Qwen3.8 quantization recipe.

## Expected Paths

- `flake.nix`
- `CMakeLists.txt`
- `tools/strix-m2-gate.py`
- `tests/tools/test_m2_gate.py`
- `docs/PROJECT_STATUS.md`
- `docs/ROADMAP.md`

## Definition of Done

- [ ] One Nix command runs native T0/T1 tests, Python inspection/tokenizer/artifact/comparison/SHQ/eval tests, and fails on any skipped mandatory component.
- [ ] JSON attestation maps exits: deterministic malformed safetensors rejection; pinned tokenizer/template parity; byte-exact CPU SHQ vectors; teacher-forced full-vocabulary capture/compare; source-storage versus accumulation dtype distinction.
- [ ] The gate uses only committed small fixtures by default and documents an explicit artifact-backed extended command for Qwen3.5-0.8B teacher capture without making local weights a pass requirement.
- [ ] Attestation records test revision and fixture/schema hashes without timestamps in reproducibility identity.
- [ ] Status documentation says SHQ remains candidate and lists Milestone 3 CPU/HIP/AIE/shared-layout evidence as subsequent work.

## Development Loop

```sh
nix develop -c python3 -m unittest -v tests.tools.test_m2_gate
nix develop -c bash -lc 'python3 tools/strix-m2-gate.py --json artifacts/m2-gate.json && python3 -m json.tool artifacts/m2-gate.json >/dev/null'
git add -A
nix build -L
```

## Output Artifacts

- Machine-readable Milestone 2 exit attestation
- Single documented Nix gate
- Fixture and schema hash inventory

## Stop Conditions

- Stop if any exit is attested only by prose or a skipped test.
- Stop if the default gate needs network access, model weights, GPU, or NPU.
- Stop if the gate claims accelerator parity or SHQ v1 freeze.

## ROADMAP Traceability

- M2 exits 1-5
- ROADMAP Delivery Rules
- TESTING.md Pull Request
- PROJECT_STATUS.md Current Implementation Boundary

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
