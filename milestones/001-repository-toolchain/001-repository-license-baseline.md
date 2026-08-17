---
id: M001-C001
title: "Normalize the repository and MIT license baseline"
milestone: M001
status: planned
dependencies: []
---

# M001-C001: Normalize the repository and MIT license baseline

## Dependencies

- None.

## Required Context

- docs/ROADMAP.md
- README.md
- AGENTS.md
- LICENSE
- .gitignore

## Goal

Leave a clean, intentionally initialized source repository with the complete canonical MIT license and documented Nix-only contribution boundary.

## Non-Goals

- Do not create the native source/model/test directory tree; M001-C003 owns it.
- Do not add third-party attribution; M001-C002 owns it.
- Do not add CMake presets, checks, or CI entry points.

## Expected Paths

- `LICENSE`
- `README.md`
- `AGENTS.md`
- `.gitignore`

## Definition of Done

- [ ] LICENSE contains the complete unmodified MIT grant, condition, and warranty disclaimer with the project copyright line.
- [ ] README.md and AGENTS.md consistently state Linux x86-64/Strix Halo support and Nix-only build policy without advertising Makefile or direct host builds.
- [ ] Generated CMake, Nix result, test-output, and model-artifact paths are ignored while source and fixture paths remain trackable.
- [ ] The PR contains only repository/license baseline changes.

## Development Loop

```sh
nix develop -c python3 -c 'from pathlib import Path; s=Path("LICENSE").read_text(); assert s.startswith("MIT License\n"); assert "Permission is hereby granted" in s; assert "THE SOFTWARE IS PROVIDED \"AS IS\"" in s'
nix flake check --no-build
```

## Output Artifacts

- Committed canonical LICENSE
- Nix flake evaluation result in the PR check log

## Stop Conditions

- Stop and fail if the intended copyright holder/year cannot be confirmed.
- Stop and fail if any proposed ignore rule would hide source, tests, fixtures, notices, or lock files.
- Stop after the license/repository baseline passes; do not absorb layout or toolchain work.

## ROADMAP Traceability

- tasks: M0 task 1
- firstBacklog: First Backlog item 1 (license portion)

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
