---
id: M002-C007
title: "Define the canonical machine fingerprint and benchmark artifact binding"
milestone: M002
status: planned
dependencies: [M002-C001, M002-C002, M001-C005, M001-C008]
---

# M002-C007: Define the canonical machine fingerprint and benchmark artifact binding

## Dependencies

- [M002-C001](001-diagnose-command-contract.md)
- [M002-C002](002-hardware-software-inventory.md)
- [M001-C005](../001-repository-toolchain/005-supported-toolchain-pins.md)
- [M001-C008](../001-repository-toolchain/008-single-pr-test-entrypoint.md)

## Required Context

- docs/ROADMAP.md
- docs/TESTING.md
- docs/PERFORMANCE.md
- docs/BENCHMARKS.md
- docs/PROJECT_STATUS.md
- docs/GPU_BACKEND.md
- docs/NPU_BACKEND.md

## Goal

Define, generate, validate, hash, and require one canonical structured machine fingerprint in every diagnostic and benchmark artifact.

## Non-Goals

- Running bandwidth or thermal measurements
- Including secrets, usernames, hostnames, serial numbers, or unstable free-memory snapshots in the identity hash
- Overwriting historical baselines
- Treating two different software/firmware stacks as the same benchmark identity

## Expected Paths

- `src/core/diagnostics/fingerprint.cpp`
- `src/core/diagnostics/fingerprint.h`
- `src/core/diagnostics/artifact_validator.cpp`
- `src/core/diagnostics/artifact_validator.h`
- `tests/diagnostics/fingerprint_test.cpp`
- `tests/fixtures/diagnostics/fingerprint_v1.json`
- `docs/BENCHMARKS.md`
- `docs/TESTING.md`
- `docs/CLI.md`

## Definition of Done

- [ ] Fingerprint v1 canonically records CPU topology/model, memory capacity/configuration where detectable, gfx1151 identity/CUs, XDNA2 identity/resources, Linux/kernel drivers, firmware, ROCm/HIP, XRT/plugin, compiler/build revision, power mode, and stable clock configuration
- [ ] Canonical serialization has fixed field names/types/order rules, explicit null/unavailable reasons, schemaVersion, and SHA-256 fingerprint ID
- [ ] Volatile observations such as current free memory, temperature, transient clocks, timestamps, paths, hostname, usernames, and device serials are excluded from the identity hash but may appear in a separate observation section
- [ ] Redaction tests prove no hostname, username, serial, environment secret, or absolute home path is emitted
- [ ] The artifact validator rejects missing fingerprint, hash mismatch, unsupported schema, incompatible architecture, and benchmark/fingerprint contradiction
- [ ] A shared artifact builder makes fingerprint binding mandatory so C003, C004, C005, and C006 cannot emit valid benchmark artifacts without it
- [ ] Tests include canonicalization stability, one-field hash changes, unavailable fields, privacy redaction, corrupted hashes, and golden schema validation
- [ ] One focused PR adds fingerprint/schema/validator plumbing and documentation only

## Development Loop

```sh
nix develop -c cmake --preset test
nix develop -c cmake --build --preset test
git add -A
nix build
nix flake check
./result/bin/strix-server diagnose --fingerprint --json --output /tmp/strix-fingerprint.json
./result/bin/strix-server diagnose --validate-artifact /tmp/strix-fingerprint.json
nix develop -c ctest --preset test --output-on-failure -R 'fingerprint|artifact_validator'
nix develop -c bash -lc './result/bin/strix-server diagnose --fingerprint --json > /tmp/fp-a.json && ./result/bin/strix-server diagnose --fingerprint --json > /tmp/fp-b.json && python -c "import json; a=json.load(open(\"/tmp/fp-a.json\")); b=json.load(open(\"/tmp/fp-b.json\")); assert a[\"fingerprintId\"] == b[\"fingerprintId\"]"'
# Repeat generation on the same unchanged host to attest stable identity despite different observation timestamps.
```

## Output Artifacts

- /tmp/strix-fingerprint.json (local machine fingerprint; retain with benchmark artifacts, do not commit host identifiers)
- tests/fixtures/diagnostics/fingerprint_v1.json (synthetic sanitized golden)
- Fingerprint schema and canonicalization rules documented in docs/BENCHMARKS.md

## Stop Conditions

- M002-C002 cannot provide an authoritative gfx1151 or XDNA2 identity
- The proposed hash includes volatile or privacy-sensitive fields
- Canonical serialization differs across two runs on an unchanged host
- Existing benchmark writers cannot be made to fail closed when fingerprint binding is absent
- Schema changes would silently reinterpret existing artifact IDs

## ROADMAP Traceability

- docs/ROADMAP.md Milestone 1 task 7
- docs/ROADMAP.md Milestone 1 exit criteria 1, 3, and 4
- docs/TESTING.md#baseline-and-artifact-management
- docs/PERFORMANCE.md#measurement-rules
- docs/BENCHMARKS.md#reporting-rules

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
