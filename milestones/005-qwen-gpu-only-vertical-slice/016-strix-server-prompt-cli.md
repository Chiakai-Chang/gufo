---
id: M005-C016
title: "Expose deterministic generation as strix-server prompt"
milestone: M005
status: planned
dependencies: [M005-C015]
---

# M005-C016: Expose deterministic generation as strix-server prompt

## Dependencies

- [M005-C015](015-direct-generation-request.md)

## Required Context

- The documented one-shot surface accepts one of --prompt, a positional prompt, --prompt-file, or --stdin. Existing names and exit codes must remain stable.

## Goal

Implement direct greedy strix-server prompt with exact input-source validation, text/JSON output, stderr diagnostics, SIGINT cancellation, and documented exit-code mapping.

## Non-Goals

- strix-server chat
- strix-server serve
- --connect client mode
- Sampling flags beyond rejecting conflicts
- JSONL streaming
- Schema constraints

## Expected Paths

- read: `docs/CLI.md`
- write: `src/cli/prompt.*`
- write: `src/cli/main.*`
- write: `tests/cli/prompt_cli_test.*`
- write: `tests/fixtures/cli/`

## Definition of Done

- [ ] Named, positional, file, and stdin prompt sources work; zero or multiple sources fail with exit 2.
- [ ] --greedy is canonical and conflicting temperature/top-p/top-k/seed options fail with exit 2.
- [ ] Success, capability unavailable, backend failure, corrupt artifact, and interrupt map to documented codes 0, 3, 5, 7, and 130.
- [ ] Plain stdout contains only generated text; JSON is valid; diagnostics stay on stderr and redact prompt/model secrets.
- [ ] No command alias replaces strix-server prompt.

## Development Loop

```sh
nix develop -c cmake --preset hip-test
nix develop -c cmake --build --preset hip-test
git add -A
nix build
nix develop -c ctest --preset hip-test --output-on-failure -R 'cli_prompt_(inputs|outputs|exit_codes|redaction)'
```

## Hardware Validation

```sh
mkdir -p artifacts/m005
nix develop -c bash -lc './result/bin/strix-server prompt --config tests/fixtures/cli/qwen08b.toml --model qwen35-08b --greedy --max-output-tokens 4 --prompt Hello --output json > artifacts/m005/prompt-smoke.json'
```

## Output Artifacts

- CLI input/output golden fixtures
- gfx1151 terminal prompt smoke JSON

## Stop Conditions

- Stop if implementing prompt requires renaming the command or creating a second inference path.
- Stop if machine-readable stdout contains diagnostics or terminal control bytes.

## ROADMAP Traceability

- tasks: M4.7 strix-server prompt in direct greedy mode
- deferred: M4.9 sampling/chat receives no completion credit in this greedy card
- exitCriteria: Safetensors-derived artifact produces text from terminal prompt
- docs: docs/CLI.md#One-Shot-Prompts
- docs: docs/CLI.md#Exit-Codes

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
