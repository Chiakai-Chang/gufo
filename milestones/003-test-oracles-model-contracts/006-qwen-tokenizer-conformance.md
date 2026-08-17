---
id: M003-C006
title: "Implement exact Qwen tokenizer conformance"
milestone: M003
status: planned
dependencies: [M003-C004, M003-C005]
---

# M003-C006: Implement exact Qwen tokenizer conformance

## Dependencies

- [M003-C004](004-qwen35-08b-descriptor.md)
- [M003-C005](005-qwen38-27b-text-descriptor-reuse.md)

## Required Context

- Token IDs are part of numerical identity. Both descriptors must name validated tokenizer assets; native code must not execute arbitrary tokenizer plugins.

## Goal

Implement the curated native Qwen tokenizer path and a committed corpus of source-generated token IDs shared by Qwen3.5-0.8B and Qwen3.8-27B where their pinned tokenizer identity matches.

## Non-Goals

- Do not implement chat rendering; that is M003-C007.
- Do not implement sampling, stop matching, HTTP adapters, or a generic tokenizer runtime.
- Do not regenerate goldens in CI.

## Expected Paths

- `src/tokenization/qwen_tokenizer.hpp`
- `src/tokenization/qwen_tokenizer.cpp`
- `tests/tokenization/qwen_tokenizer_test.cpp`
- `tests/fixtures/tokenization/qwen-corpus.json`
- `tools/strix-tokenizer-fixture.py`
- `CMakeLists.txt`

## Definition of Done

- [ ] Committed cases cover empty text, ASCII, whitespace, Unicode, combining characters, byte fallback/invalid-byte policy, added/special tokens, BOS/EOS options, and long-token boundaries.
- [ ] Fixture metadata records source repository/revision, tokenizer asset hashes, Transformers/tokenizers versions, invocation, and expected IDs.
- [ ] Native encoding matches every pinned source ID exactly and deterministic decode/encode behavior is tested.
- [ ] Fixture regeneration is an explicit Python tool under the existing Nix Python environment and comparison tests never contact the network.

## Development Loop

```sh
nix develop -c bash -lc 'cmake -S . -B build/m003-c006 -DBUILD_TESTING=ON && cmake --build build/m003-c006 --target qwen_tokenizer_test && ctest --test-dir build/m003-c006 --output-on-failure -R ^qwen_tokenizer_test$'
nix develop -c python3 -m unittest -v tests.tools.test_tokenizer_fixture_schema
```

## Output Artifacts

- Committed tokenizer conformance corpus
- Explicit offline fixture-generation utility

## Stop Conditions

- Stop if goldens lack immutable tokenizer asset hashes and source revision.
- Stop if tests fetch from Hugging Face or regenerate expected IDs automatically.
- Stop if arbitrary tokenizer code from model artifacts is required.

## ROADMAP Traceability

- M2 task 5
- M2 exit 2
- First Backlog item 7
- TOKENIZATION.md Supported Tokenizers

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
