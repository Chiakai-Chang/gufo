---
id: M001-C002
title: "Add project NOTICE and initial third-party inventory"
milestone: M001
status: done
dependencies: [M001-C001]
---

# M001-C002: Add project NOTICE and initial third-party inventory

## Dependencies

- [M001-C001](001-repository-license-baseline.md)

## Required Context

- docs/ROADMAP.md
- docs/LICENSING.md
- LICENSE
- flake.lock
- flake.nix
- .devops/nix/package.nix
- .devops/nix/xrt.nix
- .devops/nix/xrt-plugin-amdxdna.nix

## Goal

Create reviewable NOTICE and THIRD_PARTY_NOTICES.md files that identify every dependency currently linked, copied, patched, bundled, or required from the system.

## Non-Goals

- Do not generate a full release SBOM.
- Do not change dependency revisions or packaging.
- Do not make legal conclusions beyond recording provenance and obligations.

## Expected Paths

- `NOTICE`
- `THIRD_PARTY_NOTICES.md`

## Definition of Done

- [x] NOTICE identifies Strix-Halo.cpp and points recipients to LICENSE and THIRD_PARTY_NOTICES.md.
- [x] THIRD_PARTY_NOTICES.md records at least name, upstream URL, pinned version/revision, component used, SPDX identifier, copyright/notice source, relationship (linked/copied/modified/system), and corresponding-source location for XRT, amd/xdna-driver shim, ROCm/HIP, hipBLAS, rocBLAS, and other shipped direct dependencies.
- [x] Apache-2.0 NOTICE obligations for the exact pinned XRT and XDNA sources are checked and applicable text is preserved rather than guessed.
- [x] The inventory distinguishes the system kernel driver and firmware from distributed userspace content.
- [x] The PR contains only notice/inventory documentation.

## Development Loop

```sh
nix develop -c python3 -c 'from pathlib import Path; assert Path("NOTICE").is_file(); s=Path("THIRD_PARTY_NOTICES.md").read_text(); [(_ for _ in ()).throw(AssertionError(x)) for x in ("XRT","xdna-driver","ROCm","hipBLAS","rocBLAS") if x not in s]'
nix flake check --no-build
```

## Output Artifacts

- Committed NOTICE
- Committed THIRD_PARTY_NOTICES.md with revision-level provenance

## Stop Conditions

- Stop and mark blocked if any shipped or linked component has no identifiable license or source revision.
- Stop and mark blocked if required upstream NOTICE text cannot be obtained for the pinned source.
- Stop before changing Nix dependency pins; file discrepancies for M001-C005.

## ROADMAP Traceability

- tasks: M0 task 2

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
