---
id: M002-C001
title: "Add the strix-server diagnose command and stable output contract"
milestone: M002
status: planned
dependencies: [M001-C002, M001-C003, M001-C008]
---

# M002-C001: Add the strix-server diagnose command and stable output contract

## Dependencies

- [M001-C002](../001-repository-toolchain/002-initial-third-party-notices.md)
- [M001-C003](../001-repository-toolchain/003-cpp20-server-skeleton.md)
- [M001-C008](../001-repository-toolchain/008-single-pr-test-entrypoint.md)

## Required Context

- docs/ROADMAP.md
- docs/CLI.md
- docs/TESTING.md
- docs/PROJECT_STATUS.md
- src/main.cpp
- CMakeLists.txt
- flake.nix
- .devops/nix/package.nix

## Goal

Implement a non-interactive strix-server diagnose command that dispatches diagnostic sections, supports human and JSON output, returns documented status codes, and can be extended by the remaining Milestone 1 cards without changing its public envelope.

## Non-Goals

- Collecting the complete hardware inventory
- Running bandwidth, thermal, HIP, or XRT device benchmarks
- Adding inference, HTTP serving, chat, or model loading
- Supporting non-Linux, non-x86-64, CUDA, or generic accelerator platforms

## Expected Paths

- `src/cli/diagnose.cpp`
- `src/cli/diagnose.h`
- `src/core/diagnostics/report.cpp`
- `src/core/diagnostics/report.h`
- `tests/diagnostics/diagnose_cli_test.cpp`
- `docs/CLI.md`
- `CMakeLists.txt`

## Definition of Done

- [ ] The installed strix-server accepts `diagnose`, `diagnose --json`, and `diagnose --help` without prompts
- [ ] JSON output has a versioned top-level envelope with schemaVersion, engineRevision, timestamp, status, checks, warnings, and errors
- [ ] Unknown options and collector failures return nonzero and identify the failing collector; unsupported hardware is not reported as success
- [ ] Human output is rendered from the same report object as JSON
- [ ] CLI tests cover help, deterministic schema keys, malformed options, unavailable optional collectors, stdout/stderr separation, and exit codes
- [ ] No CUDA headers, libraries, symbols, or terminology are introduced
- [ ] One focused PR contains only the command/report framework, tests, and its CLI documentation

## Development Loop

```sh
nix develop -c cmake --preset test
nix develop -c cmake --build --preset test
git add -A
nix build
nix flake check
./result/bin/strix-server diagnose --help
./result/bin/strix-server diagnose --json > /tmp/strix-diagnose-envelope.json
nix develop -c ctest --preset test --output-on-failure -R 'diagnose_cli'
# The CTest invocation is inside `nix develop`; configure/build steps must be provided by the M001 documented Nix test entry point and must not be run directly. If M001 exposes a different installed executable path, update docs and all card commands consistently in this PR.
```

## Output Artifacts

- /tmp/strix-diagnose-envelope.json (ephemeral validation output)
- Versioned diagnose JSON schema/example committed under tests/fixtures/diagnostics/

## Stop Conditions

- M001 has not delivered an installed strix-server executable and documented Nix PR test entry point
- The proposed CLI requires renaming an existing public executable without an approved repository decision
- A JSON schema cannot represent partial collector failure without discarding successful sections
- Any implementation path introduces a CUDA dependency or a non-Nix build command

## ROADMAP Traceability

- docs/ROADMAP.md#milestone-1-hardware-diagnostics-and-baselines task 1
- docs/ROADMAP.md Milestone 1 delivery rule that every milestone remains runnable and testable
- docs/TESTING.md#pull-request-and-release-suites

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
