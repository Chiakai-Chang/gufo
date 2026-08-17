---
id: M003-C002
title: "Complete deterministic safetensors index and shard inspection"
milestone: M003
status: planned
dependencies: [M001-C008]
---

# M003-C002: Complete deterministic safetensors index and shard inspection

## Dependencies

- [M001-C008](../001-repository-toolchain/008-single-pr-test-entrypoint.md)

## Required Context

- tools/strix/safetensors.py and tools/strix-inspect.py already implement the Python source-inspection path. This card reuses and closes that implementation rather than creating a second inspector.

## Goal

Make Python snapshot inspection deterministic and fail-closed for malformed headers, duplicate/index mismatches, missing or extra tensors, unreferenced shards, invalid dtype/shape/offset arithmetic, and non-immutable publication revisions.

## Non-Goals

- Do not implement native C++ safetensors parsing.
- Do not download large checkpoints in tests.
- Do not quantize or load tensors into a model.

## Expected Paths

- `tools/strix/safetensors.py`
- `tools/strix-inspect.py`
- `tools/strix/manifest.py`
- `tests/tools/test_safetensors.py`
- `tests/fixtures/safetensors/`

## Definition of Done

- [ ] Single-file and indexed multi-shard inventories sort deterministically and produce byte-identical source manifests on repeat runs.
- [ ] Expected tensor sets are enforced in both directions and all unexpected .safetensors shards fail unless an explicit research-only option is documented.
- [ ] Malformed synthetic fixtures cover truncated length/header, invalid JSON, duplicate JSON keys, unknown dtype, invalid dimensions, byte-count overflow, overlap, gap/bounds policy, missing shard/tensor, extra shard/tensor, and index traversal attempts.
- [ ] Manifest records SHA-256, exact bytes, source storage dtype, architecture, tokenizer/template hashes, repository, and immutable revision without embedding timestamps.

## Development Loop

```sh
nix develop -c python3 -m unittest -v tests.tools.test_safetensors
nix develop -c bash -lc 'python3 tools/strix-inspect.py --help >/dev/null'
```

## Output Artifacts

- Small synthetic safetensors malformed-input fixtures
- Deterministic strix.source.v1 manifest golden

## Stop Conditions

- Stop and fail if an indexed and physical inventory can disagree without rejection.
- Stop if a test would require committing or downloading model weights.
- Stop if output identity depends on filesystem enumeration order, mtime, or wall clock.

## ROADMAP Traceability

- M2 task 2
- M2 exit 1
- First Backlog item 6
- QUANTIZATION.md Safetensors Source Contract

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
