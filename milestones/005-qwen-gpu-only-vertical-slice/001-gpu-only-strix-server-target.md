---
id: M005-C001
title: "Wire the existing strix-server target to the GPU-only inference runtime"
milestone: M005
status: planned
dependencies: [M001-C008, M003-C013, M004-C011]
---

# M005-C001: Wire the existing strix-server target to the GPU-only inference runtime

## Dependencies

- [M001-C008](../001-repository-toolchain/008-single-pr-test-entrypoint.md)
- [M003-C013](../003-test-oracles-model-contracts/013-milestone2-contract-gate.md)
- [M004-C011](../004-gpu-npu-format-feasibility/011-gpu-format-feasibility-gate.md)

## Required Context

- Milestone 0 owns executable creation and command naming. This card only wires the already-created `strix-server` target to the reusable GPU inference libraries. The existing `strix` binary remains untouched unless a separate decision card says otherwise. The M005 MVP must not depend on XRT or `amdxdna`.

## Goal

Link the existing `strix-server` placeholder to the reusable C++20/HIP gfx1151 inference runtime and add a GPU-only package/check profile without changing executable names or command topology.

## Non-Goals

- Model loading
- Token generation
- NPU discovery or execution
- Renaming documented commands
- HTTP serving

## Expected Paths

- read: `flake.nix`
- read: `.devops/nix/`
- read: `CMakeLists.txt`
- read: `src/main.cpp`
- read: `docs/CLI.md`
- read: `docs/GPU_BACKEND.md`
- write: `flake.nix`
- write: `.devops/nix/`
- write: `CMakeLists.txt`
- write: `src/server/`
- write: `tests/static/`

## Definition of Done

- [ ] `nix build` preserves the existing executables and produces `result/bin/strix-server` linked to reusable GPU runtime targets for gfx1151.
- [ ] `strix-server --version` and `--help` remain deterministic; `prompt` may report capability unavailable until later cards.
- [ ] The `strix-server` ELF dependency and symbol scans contain no CUDA, XRT, `amdxdna`, or AIE dependency.
- [ ] The existing `strix` probe/CLI behavior and installed path are unchanged.
- [ ] No model numerical implementation is duplicated in this card; it links the promoted Milestone 3 GPU targets.

## Development Loop

```sh
nix develop -c cmake --preset hip-test
nix develop -c cmake --build --preset hip-test
git add -A
nix build
nix develop -c ctest --preset hip-test --output-on-failure -R 'static_(command_name|no_cuda|no_npu)'
nix develop -c bash -lc './result/bin/strix-server --version && ./result/bin/strix-server --help | grep -F prompt'
```

## Hardware Validation

```sh
nix develop -c bash -lc 'rocminfo | grep -q gfx1151 && ./result/bin/strix-server --version'
```

## Output Artifacts

- Nix build result containing result/bin/strix-server
- Machine-readable dependency/symbol scan report

## Stop Conditions

- Stop if gfx1151 code-object compilation cannot be expressed through the pinned Nix ROCm toolchain.
- Stop if GPU-only linkage requires removing or renaming an existing executable or command; preserve topology and report the conflict.
- Stop if the card would duplicate Milestone 3 kernels rather than link their promoted targets.

## ROADMAP Traceability

- tasks: M4.7 strix-server prompt naming prerequisite
- tasks: M4.10 eager-before-graphs build boundary
- exitCriteria: Runnable terminal path prerequisite
- docs: docs/CLI.md#Commands
- docs: docs/GPU_BACKEND.md#Build
- docs: AGENTS.md#Build-(Nix)

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
