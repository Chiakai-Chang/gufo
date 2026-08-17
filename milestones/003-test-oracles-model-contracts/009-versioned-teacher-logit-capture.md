---
id: M003-C009
title: "Harden the full-quality teacher-logit capture workflow and artifact"
milestone: M003
status: planned
dependencies: [M003-C002, M003-C006, M003-C007]
---

# M003-C009: Harden the full-quality teacher-logit capture workflow and artifact

## Dependencies

- [M003-C002](002-safetensors-inspection-hardening.md)
- [M003-C006](006-qwen-tokenizer-conformance.md)
- [M003-C007](007-qwen-chat-template-contract.md)

## Required Context

- tools/strix-capture.py, tools/strix/model.py, and the zstd-enabled Nix shell already capture Qwen3.5 teacher logits. The workflow needs dynamic metadata, complete identity, validation, and explicit dtype semantics rather than replacement.

## Goal

Evolve the existing capture tool into a validated versioned teacher-forced full-vocabulary logit artifact for pinned Qwen sources and suites, with content identity and complete reproducibility metadata.

## Non-Goals

- Do not commit full logits or model weights.
- Do not capture hosted top-k-only API output.
- Do not add candidate comparison; that is M003-C010.

## Expected Paths

- `tools/strix-capture.py`
- `tools/strix/model.py`
- `tools/strix/manifest.py`
- `tests/tools/test_capture_artifact.py`
- `tests/fixtures/logits/tiny-teacher/`
- `docs/BENCHMARKS.md`

## Definition of Done

- [ ] Vocabulary size/order and tokenizer/template hashes come from validated model assets, never a hard-coded 248320 constant.
- [ ] Manifest records source repo/revision/file hashes, suite/token-stream hash, reference runtime/version, source storage dtype, accumulation dtype/settings, generation/teacher-forcing parameters, chunk shape/dtype, and artifact schema.
- [ ] All payloads including the finalized manifest are covered by checksums or a non-circular content-addressed manifest scheme, and corruption/missing/extra chunk validation fails.
- [ ] Matched-token positions and target tokens are unambiguous across prompts; repeated fixture capture is byte-identical except explicitly excluded diagnostic timing.
- [ ] Qwen3.5-0.8B remains the routine capture fixture; the schema is descriptor-driven so Qwen3.8-27B can reuse it without hard-coded 0.8B dimensions.

## Development Loop

```sh
nix develop -c python3 -m unittest -v tests.tools.test_capture_artifact
nix develop -c bash -lc 'tools/strix-capture.py --help >/dev/null'
```

## Output Artifacts

- Tiny synthetic strix.logit-artifact fixture
- Validated artifact schema
- Capture provenance report

## Stop Conditions

- Stop if vocabulary identity is inferred only from array width.
- Stop if checksums omit a required artifact file.
- Stop if storage BF16 is described as FP32 merely because logits are serialized f32.
- Stop if implementation requires committing large logits.

## ROADMAP Traceability

- M2 tasks 7-8
- M2 exit 4
- First Backlog item 8
- TESTING.md Full-Logit Testing

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
