---
id: M003-C004
title: "Define the Qwen3.5-0.8B explicit model contract"
milestone: M003
status: planned
dependencies: [M003-C001, M003-C003]
---

# M003-C004: Define the Qwen3.5-0.8B explicit model contract

## Dependencies

- [M003-C001](001-checked-tensor-descriptor.md)
- [M003-C003](003-compiled-model-kind-registry.md)

## Required Context

- Qwen3.5-0.8B is the rapid correctness fixture. tools/strix/model.py contains useful eligible tensor suffixes and teacher loading, but not a complete explicit inventory/orientation/state contract.

## Goal

Create one reviewed descriptor for the pinned Qwen3.5-0.8B revision covering dimensions, exact tensor inventory and logical roles, source-to-logical orientations, tied weights, repeating block pattern, RoPE, tokenizer/template IDs, MTP, and recurrent/KV state metadata.

## Non-Goals

- Do not implement model execution or kernels.
- Do not select a production Qwen3.8 quantization recipe from 0.8B evidence.
- Do not include vision tensors in the serving tensor inventory.

## Expected Paths

- `models/qwen35_08b/qwen35_descriptor.hpp`
- `models/qwen35_08b/qwen35_08b_descriptor.cpp`
- `tools/strix/model.py`
- `tests/models/qwen35_08b_descriptor_test.cpp`
- `tests/fixtures/models/qwen35-08b-config.json`
- `tests/fixtures/models/qwen35-08b-inventory.json`

## Definition of Done

- [ ] Every expected text/MTP source tensor maps once to a stable logical role and declared orientation; missing, extra, ambiguous, or shape-incompatible tensors fail.
- [ ] The descriptor explicitly encodes the 3 Gated DeltaNet plus 1 full-attention repeating pattern and pinned dimensions/config values.
- [ ] Vision source names are classified as excluded-by-text-capability, never silently interpreted as text tensors.
- [ ] Python inspection and native descriptor tests consume the same committed inventory fixture, avoiding duplicate hand-maintained eligibility lists.

## Development Loop

```sh
nix develop -c bash -lc 'cmake -S . -B build/m003-c004 -DBUILD_TESTING=ON && cmake --build build/m003-c004 --target qwen35_08b_descriptor_test && ctest --test-dir build/m003-c004 --output-on-failure -R ^qwen35_08b_descriptor_test$'
nix develop -c python3 -m unittest -v tests.tools.test_qwen_descriptor
```

## Output Artifacts

- Pinned Qwen3.5-0.8B config fixture
- Complete expected inventory/orientation fixture

## Stop Conditions

- Stop if any released tensor role is obtained by generic name guessing.
- Stop if vision is accepted into the text serving inventory.
- Stop if source revision or dimensions are not pinned.

## ROADMAP Traceability

- M2 task 4
- ROADMAP Objective
- PROJECT_STATUS.md Models
- QUANTIZATION.md Model-Specific Import Adapters

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
