---
id: M001-C004
title: "Add development, release, sanitizer, and test presets"
milestone: M001
status: planned
dependencies: [M001-C003]
---

# M001-C004: Add development, release, sanitizer, and test presets

## Dependencies

- [M001-C003](003-cpp20-server-skeleton.md)

## Required Context

- docs/ROADMAP.md
- docs/TESTING.md
- CMakeLists.txt
- flake.nix
- .devops/nix/package.nix
- .gitignore

## Goal

Provide reproducible CMake configure/build presets for development, release, sanitizer, and test workflows, all entered through the Nix development environment.

## Non-Goals

- Do not make direct host CMake an officially supported entry point.
- Do not add lint, documentation, dependency, or CUDA scans.
- Do not require GPU/NPU hardware for sanitizer and unit-test presets.

## Expected Paths

- `CMakePresets.json`
- `CMakeLists.txt`
- `.gitignore`
- `README.md`

## Definition of Done

- [ ] Versioned CMakePresets.json defines configure/build/test presets named development, release, sanitizer, test, hip-test, and npu-test with separate ignored binary directories.
- [ ] Development enables warnings and debug information; release uses optimized release semantics; sanitizer enables AddressSanitizer and UndefinedBehaviorSanitizer in a CPU-only configuration; test enables CPU-only CTest; hip-test enables HIP for gfx1151 without XRT; npu-test enables the pinned HIP/XRT research stack.
- [ ] Each preset configures and builds the existing executables and libraries with Ninja from `nix develop`; adding `strix-server` must not remove or silently rename the existing `strix` target.
- [ ] The test preset executes CPU tests without hardware. Hardware presets label tests so unavailable devices skip normally and fail under their documented `STRIX_REQUIRE_*` mode.
- [ ] README.md documents Nix-wrapped preset usage and does not present raw host CMake as supported.

## Development Loop

```sh
nix develop -c cmake --preset development
nix develop -c cmake --build --preset development
nix develop -c cmake --preset release
nix develop -c cmake --build --preset release
nix develop -c cmake --preset sanitizer
nix develop -c cmake --build --preset sanitizer
nix develop -c cmake --preset test
nix develop -c cmake --build --preset test
nix develop -c ctest --preset test --output-on-failure
nix develop -c cmake --preset hip-test
nix develop -c cmake --build --preset hip-test
nix develop -c cmake --preset npu-test
nix develop -c cmake --build --preset npu-test
```

## Output Artifacts

- CMakePresets.json
- Ignored build/development, build/release, build/sanitizer, and build/test trees
- CTest --version smoke result

## Stop Conditions

- Stop and fail if preset inheritance makes sanitizer or tests silently use release flags.
- Stop and fail if the sanitizer preset links HIP/XRT device runtimes or requires Strix Halo hardware.
- Stop before adding project-wide quality checks; M001-C007 owns them.

## ROADMAP Traceability

- tasks: M0 task 4
- firstBacklog: First Backlog item 2 (CMake presets portion)

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
