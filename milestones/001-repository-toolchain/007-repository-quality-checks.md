---
id: M001-C007
title: "Add formatting, static-analysis, dependency, and documentation checks"
milestone: M001
status: planned
dependencies: [M001-C002, M001-C003, M001-C004, M001-C005, M001-C006]
---

# M001-C007: Add formatting, static-analysis, dependency, and documentation checks

## Dependencies

- [M001-C002](002-initial-third-party-notices.md)
- [M001-C003](003-cpp20-server-skeleton.md)
- [M001-C004](004-cmake-build-presets.md)
- [M001-C005](005-supported-toolchain-pins.md)
- [M001-C006](006-reject-cuda-contamination.md)

## Required Context

- docs/ROADMAP.md
- docs/TESTING.md
- docs/LICENSING.md
- CMakeLists.txt
- CMakePresets.json
- flake.nix
- .devops/nix/package.nix
- src/
- tools/
- docs/

## Goal

Expose deterministic, non-mutating Nix checks for C/C++ formatting, C/C++ static analysis, dependency/license inventory consistency, and Markdown/link integrity.

## Non-Goals

- Do not create the single PR umbrella command; M001-C008 owns orchestration and documentation.
- Do not auto-format or rewrite documentation in validation mode.
- Do not lint generated artifacts, model weights, benchmark outputs, or vendored upstream source trees.

## Expected Paths

- `.clang-format`
- `.clang-tidy`
- `cmake/StaticAnalysis.cmake`
- `tools/check-dependencies.py`
- `tools/check-docs.py`
- `flake.nix`
- `.devops/nix/package.nix`
- `tests/static/`

## Definition of Done

- [ ] The format check runs pinned clang-format in dry-run/error mode over tracked C/C++ files.
- [ ] The static-analysis check runs pinned clang-tidy against an exported compilation database with an explicit stable rule set and zero findings for the skeleton.
- [ ] The dependency check emits a deterministic direct/runtime inventory and fails when a shipped direct dependency lacks an entry in THIRD_PARTY_NOTICES.md or when forbidden CUDA dependencies appear.
- [ ] The documentation check validates local Markdown links, anchors, fenced JSON examples, and required roadmap/licensing files without network access.
- [ ] Each check is an independently addressable flake check and succeeds from a clean checkout.
- [ ] Check outputs are concise on success and actionable with file and line on failure.

## Development Loop

```sh
git add -A
nix build .#checks.x86_64-linux.format
git add -A
nix build .#checks.x86_64-linux.static-analysis
git add -A
nix build .#checks.x86_64-linux.dependency-inventory
git add -A
nix build .#checks.x86_64-linux.docs
```

## Output Artifacts

- Four Nix check derivations
- Deterministic dependency inventory report
- Formatting, clang-tidy, and documentation validation logs

## Stop Conditions

- Stop and fail if any check mutates the working tree.
- Stop and fail if checks download network content or use unpinned host tools.
- Stop and fail if dependency inventory and THIRD_PARTY_NOTICES.md disagree.
- Stop after independent checks exist; do not add the PR umbrella in this card.

## ROADMAP Traceability

- tasks: M0 task 7
- exitCriteria: The binary and dependency scan contain no CUDA dependency (dependency-inventory reinforcement)

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
