---
id: M003-C007
title: "Implement the bounded Qwen chat template contract"
milestone: M003
status: planned
dependencies: [M003-C006]
---

# M003-C007: Implement the bounded Qwen chat template contract

## Dependencies

- [M003-C006](006-qwen-tokenizer-conformance.md)

## Required Context

- Exact tokenizer matching is insufficient if prompt rendering drifts. The server may interpret only a small validated/compiled template instruction set, not arbitrary Jinja downloaded with weights.

## Goal

Implement the Qwen text chat formatter and committed source-reference fixtures for supported role sequences, generation prompt, thinking framing, tools-as-rendered-text metadata, and special-token insertion.

## Non-Goals

- Do not execute arbitrary Jinja/Python.
- Do not implement HTTP item adapters or tool execution.
- Do not implement generation or evaluate model answers.

## Expected Paths

- `src/tokenization/qwen_chat_template.hpp`
- `src/tokenization/qwen_chat_template.cpp`
- `tests/tokenization/qwen_chat_template_test.cpp`
- `tests/fixtures/tokenization/qwen-chat-corpus.json`
- `tools/strix-template-fixture.py`

## Definition of Done

- [ ] Supported system/developer/user/assistant/tool-definition/tool-call/tool-result sequences render deterministic bytes and exact token IDs matching the pinned source template.
- [ ] Unsupported roles, malformed tool pairing, recursion/unbounded constructs, unknown special tokens, and excessive expansion fail closed.
- [ ] Template source hash and compiled representation hash are checked against each model descriptor.
- [ ] Fixtures include empty content, Unicode, delimiter-like user text, generation prompt, and thinking enabled/disabled cases.

## Development Loop

```sh
nix develop -c bash -lc 'cmake -S . -B build/m003-c007 -DBUILD_TESTING=ON && cmake --build build/m003-c007 --target qwen_chat_template_test && ctest --test-dir build/m003-c007 --output-on-failure -R ^qwen_chat_template_test$'
nix develop -c python3 -m unittest -v tests.tools.test_template_fixture_schema
```

## Output Artifacts

- Committed chat-rendering/token fixtures
- Bounded compiled-template contract

## Stop Conditions

- Stop if runtime execution of downloaded Jinja or code is introduced.
- Stop if rendered-byte and token-ID references disagree.
- Stop before transport/API formatting work.

## ROADMAP Traceability

- M2 tasks 4-5
- M2 exit 2
- TOKENIZATION.md Chat Templates
- EVAL.md Generation Contract

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
