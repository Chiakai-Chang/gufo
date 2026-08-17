---
id: M001-C003
title: "Create the C++20 source layout and placeholder strix-server"
milestone: M001
status: planned
dependencies: [M001-C001]
---

# M001-C003: Create the C++20 source layout and placeholder strix-server

## Dependencies

- [M001-C001](001-repository-license-baseline.md)

## Required Context

- docs/ROADMAP.md
- docs/CLI.md
- docs/PROJECT_STATUS.md
- CMakeLists.txt
- src/main.cpp
- .devops/nix/package.nix

## Goal

Establish the roadmap's top-level C++20 layout and build a side-effect-free placeholder strix-server whose --version command succeeds without probing GPU or NPU hardware.

## Non-Goals

- Do not implement diagnose, model loading, prompt, chat, or HTTP serving.
- Do not add build presets; M001-C004 owns them.
- Do not redesign the existing HIP/XRT probe beyond isolating it from --version and preserving it for Milestone 1 work.

## Expected Paths

- `CMakeLists.txt`
- `src/core/`
- `src/server/`
- `src/cli/`
- `src/main.cpp`
- `models/qwen35_08b/cpu/`
- `models/qwen35_08b/gpu/gfx1151/`
- `models/qwen35_08b/npu/aie2p/`
- `models/qwen38_27b/cpu/`
- `models/qwen38_27b/gpu/gfx1151/`
- `models/qwen38_27b/npu/aie2p/`
- `tests/`
- `tools/`
- `docs/`
- `.devops/nix/package.nix`

## Definition of Done

- [ ] CMake declares C++20, disables compiler extensions, and defines the installed executable target strix-server.
- [ ] All directories in docs/ROADMAP.md's Milestone 0 initial layout exist in a trackable form without placeholder production code masquerading as implementation.
- [ ] strix-server --version writes a stable project/version line, exits 0, and performs no HIP/XRT discovery or allocation.
- [ ] Unknown top-level options return the CLI/configuration error code 2, consistent with docs/CLI.md.
- [ ] The Nix package installs the new `result/bin/strix-server` placeholder without removing or renaming the existing `result/bin/strix`; `mainProgram` remains unchanged unless a separate reviewed decision updates it.
- [ ] A CPU-disabled-backend build and --version smoke pass through Nix.

## Development Loop

```sh
nix develop -c cmake -S . -B build/m001-c003 -G Ninja -DENGINE_ENABLE_HIP=OFF -DENGINE_ENABLE_XRT=OFF -DCMAKE_BUILD_TYPE=Debug
nix develop -c cmake --build build/m001-c003 --target strix-server
nix develop -c ./build/m001-c003/strix-server --version
git add -A
nix build .#default
```

## Output Artifacts

- build/m001-c003/strix-server (ignored local validation artifact)
- result/bin/strix-server from the Nix package
- Committed roadmap directory skeleton

## Stop Conditions

- Stop and fail if --version touches a device, depends on device presence, or emits probe output.
- Stop and report if adding `strix-server` requires removing, renaming, or changing the behavior of the existing `strix` binary; executable topology is outside this card.
- Stop after placeholder CLI/layout completion; diagnose belongs to Milestone 1.

## ROADMAP Traceability

- tasks: M0 task 3
- exitCriteria: A placeholder strix-server --version runs
- firstBacklog: First Backlog item 1 (repository skeleton portion)
- firstBacklog: First Backlog item 2 (strix-server --version portion)

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
