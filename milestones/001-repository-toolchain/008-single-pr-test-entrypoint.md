---
id: M001-C008
title: "Create and document the initial PR test command"
milestone: M001
status: planned
dependencies: [M001-C004, M001-C006, M001-C007]
---

# M001-C008: Create and document the initial PR test command

## Dependencies

- [M001-C004](004-cmake-build-presets.md)
- [M001-C006](006-reject-cuda-contamination.md)
- [M001-C007](007-repository-quality-checks.md)

## Required Context

- docs/ROADMAP.md
- docs/TESTING.md
- README.md
- AGENTS.md
- flake.nix
- .devops/nix/package.nix
- CMakePresets.json

## Goal

Provide one documented Nix command that builds the supported package and runs every Milestone 0 gate, including side-effect-free strix-server --version.

## Non-Goals

- Do not add hosted-CI-provider configuration; the command must be provider-neutral.
- Do not add Milestone 1 hardware diagnostics or require live GPU/NPU execution.
- Do not duplicate individual check implementations from earlier cards.

## Expected Paths

- `flake.nix`
- `.devops/nix/package.nix`
- `README.md`
- `AGENTS.md`
- `docs/TESTING.md`

## Definition of Done

- [ ] The canonical command is exactly `nix build .#checks.x86_64-linux.pr` and is documented identically in README.md, AGENTS.md, and docs/TESTING.md.
- [ ] The pr derivation composes clean configure/build, test/CTest, sanitizer tests, format, static analysis, dependency inventory, documentation validation, no-CUDA source/link/symbol/closure scans, and installed strix-server --version smoke.
- [ ] The PR check consumes the built Nix package and does not rebuild through Makefile, direct host tools, or network downloads.
- [ ] The check runs without access to HIP/XRT hardware while still building the supported gfx1151/XRT package; --version is side-effect-free.
- [ ] A clean checkout with an unchanged flake.lock can execute the documented command successfully on the supported x86_64-linux toolchain.
- [ ] The successful output retains enough derivation/log identity to attest which checks and package revision ran.

## Development Loop

```sh
git add -A
nix build .#checks.x86_64-linux.pr
nix develop -c ./result/bin/strix-server --version
```

## Output Artifacts

- Nix checks.x86_64-linux.pr derivation
- PR check log enumerating all composed gates
- Installed result/bin/strix-server version output

## Stop Conditions

- Stop and fail if the umbrella reports success while any constituent check is skipped or allowed to fail.
- Stop and fail if --version initializes HIP/XRT or requires device nodes.
- Stop and fail if the final documented clean-checkout PR entry point requires a manual `git add`, Makefile, mutable lock update, network access, or tools outside Nix. Staging this card's newly created implementation files before a development-time Nix build is allowed because Nix evaluates tracked/staged sources.
- Stop when the single entry point passes; do not begin Milestone 1 diagnostics.

## ROADMAP Traceability

- tasks: M0 task 8
- exitCriteria: A clean checkout configures and builds on the supported Strix Halo Linux environment
- exitCriteria: A placeholder strix-server --version runs
- exitCriteria: The binary and dependency scan contain no CUDA dependency
- exitCriteria: The PR test command succeeds from one documented entry point

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
