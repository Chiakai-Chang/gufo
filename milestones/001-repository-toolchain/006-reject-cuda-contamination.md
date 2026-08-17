---
id: M001-C006
title: "Reject CUDA at source, link, symbol, and runtime-closure boundaries"
milestone: M001
status: planned
dependencies: [M001-C003, M001-C005]
---

# M001-C006: Reject CUDA at source, link, symbol, and runtime-closure boundaries

## Dependencies

- [M001-C003](003-cpp20-server-skeleton.md)
- [M001-C005](005-supported-toolchain-pins.md)

## Required Context

- docs/ROADMAP.md
- docs/PROJECT_STATUS.md
- docs/GPU_BACKEND.md
- docs/TESTING.md
- CMakeLists.txt
- flake.nix
- .devops/nix/package.nix
- src/

## Goal

Make CUDA contamination a build failure by scanning build inputs and the produced strix-server binary/closure for forbidden headers, libraries, and symbols.

## Non-Goals

- Do not reject documentation that explains CUDA is unsupported.
- Do not reject ROCm compatibility macros solely by substring without proving a CUDA dependency.
- Do not add CUDA compatibility or hipify-generated source support.

## Expected Paths

- `cmake/CheckNoCuda.cmake`
- `tools/check-no-cuda.py`
- `CMakeLists.txt`
- `.devops/nix/package.nix`
- `tests/static/`

## Definition of Done

- [ ] A build-time target scans compiled source lists, include paths, compile commands, link commands, and Nix references for CUDA headers/toolkit paths and libcudart, libcublas, libcuda, or equivalent dependencies.
- [ ] Post-link scans inspect dynamic dependencies and symbol tables of strix-server and fail on CUDA libraries or API symbols.
- [ ] The Nix closure/reference scan fails if a CUDA toolkit/runtime is pulled into the shipped server closure.
- [ ] Negative fixtures containing a forbidden CUDA include, link token, and symbol prove the scanner fails; a HIP-only fixture proves it does not flag supported ROCm usage.
- [ ] The check runs during the Nix package check phase, not only as an optional developer command.

## Development Loop

```sh
nix develop -c cmake --preset test
nix develop -c cmake --build --preset test --target check-no-cuda
nix develop -c ctest --preset test -R no-cuda --output-on-failure
git add -A
nix build .#default
```

## Output Artifacts

- Machine-readable no-CUDA scan report from the test build
- Post-link dependency and symbol scan log
- Negative and positive scanner fixtures

## Stop Conditions

- Stop and fail immediately on any CUDA header, toolkit path, runtime library, exported/imported API symbol, or Nix runtime reference.
- Stop and request review if a ROCm-provided compatibility name creates a false positive; narrow the rule with an explicit fixture rather than allowlisting a broad path.
- Stop before adding unrelated formatting or documentation checks.

## ROADMAP Traceability

- tasks: M0 task 6
- exitCriteria: The binary and dependency scan contain no CUDA dependency

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
