---
id: M001-C005
title: "Pin and publish the supported Strix Halo toolchain matrix"
milestone: M001
status: planned
dependencies: [M001-C001]
---

# M001-C005: Pin and publish the supported Strix Halo toolchain matrix

## Dependencies

- [M001-C001](001-repository-license-baseline.md)

## Required Context

- docs/ROADMAP.md
- docs/PROJECT_STATUS.md
- docs/GPU_BACKEND.md
- docs/NPU_BACKEND.md
- docs/NPU_RESEARCH.md
- flake.nix
- flake.lock
- .devops/nix/package.nix
- .devops/nix/xrt.nix
- .devops/nix/xrt-plugin-amdxdna.nix

## Goal

Make the one supported Linux/compiler/ROCm/XRT/amdxdna/firmware/AIE toolchain combination explicit, immutable where Nix can pin it, and machine-checkable.

## Non-Goals

- Do not claim Nix can install or pin the host kernel driver or firmware when it cannot.
- Do not add support for additional operating systems, GPU architectures, or compatibility ranges.
- Do not implement hardware diagnosis or unsupported-version runtime rejection; those belong to Milestone 1.

## Expected Paths

- `flake.lock`
- `flake.nix`
- `.devops/nix/package.nix`
- `.devops/nix/xrt.nix`
- `.devops/nix/xrt-plugin-amdxdna.nix`
- `docs/SUPPORTED_TOOLCHAIN.md`

## Definition of Done

- [ ] docs/SUPPORTED_TOOLCHAIN.md gives exact tested identifiers for Linux distribution/kernel, C/C++ and HIP compiler, ROCm, XRT commit, amd/xdna-driver commit/plugin, firmware, and AIE compiler/tool artifacts, plus gfx1151 and XDNA2/AIE2P targets.
- [ ] Nix inputs and fetchers use immutable revisions and hashes; no branch name, floating tag, /usr fallback, or host PATH selects a build dependency.
- [ ] The documented values are derived from one machine-readable Nix/toolchain source or checked against it to prevent drift.
- [ ] Host-only kernel/firmware requirements are clearly labeled as validation prerequisites rather than Nix store outputs.
- [ ] Nix evaluation exposes the compiler and runtime versions used by the package and remains locked to x86_64-linux.

## Development Loop

```sh
nix flake metadata --json --no-write-lock-file
nix develop -c bash -c 'c++ --version && cmake --version && ninja --version'
git add -A
nix build .#default
```

## Output Artifacts

- docs/SUPPORTED_TOOLCHAIN.md
- Pinned flake.lock and fixed-output source hashes
- Toolchain-version evidence in the PR check log

## Stop Conditions

- Stop and request a maintainer decision if the supported kernel, firmware, or AIE compiler identity is unknown; do not invent a version.
- Stop and fail if any build-time dependency resolves from ambient /usr or PATH outside the Nix environment.
- Stop before implementing runtime diagnostics or broad version ranges.

## ROADMAP Traceability

- tasks: M0 task 5
- exitCriteria: A clean checkout configures and builds on the supported Strix Halo Linux environment (toolchain-definition portion)

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
