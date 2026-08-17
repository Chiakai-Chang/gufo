---
id: M005-C004
title: "Implement the native pinned Qwen tokenizer and text prompt formatter"
milestone: M005
status: planned
dependencies: [M003-C006, M003-C007, M003-C013, M005-C003]
---

# M005-C004: Implement the native pinned Qwen tokenizer and text prompt formatter

## Dependencies

- [M003-C006](../003-test-oracles-model-contracts/006-qwen-tokenizer-conformance.md)
- [M003-C007](../003-test-oracles-model-contracts/007-qwen-chat-template-contract.md)
- [M003-C013](../003-test-oracles-model-contracts/013-milestone2-contract-gate.md)
- [M005-C003](003-transactional-text-artifact-loader.md)

## Required Context

- Tokenization and prompt rendering are part of exact-token identity. Curated code must validate downloaded tokenizer data rather than execute arbitrary templates.

## Goal

Integrate the validated Milestone 2 tokenizer and bounded template implementations with the loaded runtime artifact and direct-generation request without creating a second tokenizer or formatter.

## Non-Goals

- Generic Jinja
- Tool calls
- JSON schema constraints
- Interactive chat
- HTTP adapters

## Expected Paths

- read: `docs/TOKENIZATION.md`
- read: `docs/CLI.md`
- read: `tools/suites/teacher.json`
- read: `src/tokenization/qwen_tokenizer.*`
- read: `src/tokenization/qwen_chat_template.*`
- read: `tests/fixtures/tokenization/qwen-corpus.json`
- read: `tests/fixtures/tokenization/qwen-chat-corpus.json`
- write: `models/qwen35_08b/runtime/text_binding.*`
- write: `tests/integration/qwen35_08b_text_binding_test.*`

## Definition of Done

- [ ] Runtime binding consumes the Milestone 2 tokenizer/template public APIs and committed golden corpus without redefining encode/decode behavior.
- [ ] Loaded tokenizer/template hashes, vocabulary size, and special-token IDs must match the artifact and compiled model descriptor before prompt admission.
- [ ] Raw prompt formatting and incremental decoding through the runtime binding reproduce the Milestone 2 golden fixtures exactly.
- [ ] Input limits, invalid bytes, and identity mismatches fail before GPU execution and without unbounded allocation.
- [ ] Tests prove that no alternative tokenizer or template path exists in the direct generation runtime.

## Development Loop

```sh
nix develop -c cmake --preset test
nix develop -c cmake --build --preset test
git add -A
nix build
nix develop -c ctest --preset test --output-on-failure -R 'qwen35_(tokenizer|template|incremental_decode)'
nix develop -c ctest --preset test --output-on-failure -R 'qwen35_text_binding_identity'
```

## Hardware Validation

No accelerator is required. This binding must pass under the CPU-only `test`
preset before model execution; GPU validation belongs to downstream cards.

## Output Artifacts

- Runtime text-binding identity report.
- Artifact/tokenizer/template mismatch fixtures.

## Stop Conditions

- Stop if integration requires copying or changing the Milestone 2 tokenizer/template implementation.
- Stop rather than adding normalization or special-token behavior not required by the source tokenizer.

## ROADMAP Traceability

- tasks: M4.7 prompt input prerequisite
- tasks: M4.8 exact-token fixture prerequisite
- exitCriteria: Terminal text is derived from exact tokens
- exitCriteria: Deterministic token history prerequisite
- docs: docs/TOKENIZATION.md
- docs: docs/TESTING.md#T4:-Full-model-deterministic-tests

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
