---
id: M002-C002
title: "Collect and validate the Strix Halo hardware and software inventory"
milestone: M002
status: planned
dependencies: [M002-C001, M001-C005]
---

# M002-C002: Collect and validate the Strix Halo hardware and software inventory

## Dependencies

- [M002-C001](001-diagnose-command-contract.md)
- [M001-C005](../001-repository-toolchain/005-supported-toolchain-pins.md)

## Required Context

- docs/ROADMAP.md
- docs/PROJECT_STATUS.md
- docs/PERFORMANCE.md
- docs/GPU_BACKEND.md
- docs/NPU_BACKEND.md
- docs/NPU_RESEARCH.md
- src/main.cpp
- .devops/nix/xrt.nix
- .devops/nix/xrt-plugin-amdxdna.nix

## Goal

Populate diagnose with normalized CPU, memory, gfx1151 GPU, XDNA2 NPU, kernel driver, ROCm/HIP, XRT, firmware, clock, thermal-sensor, and power-mode inventory, including explicit compatibility verdicts and actionable failures.

## Non-Goals

- Measuring throughput or latency
- Executing HIP kernels or XRT programs beyond metadata queries
- Changing pinned toolchain versions
- Supporting unknown GPUs or NPUs as production targets
- Silently accepting gfx1100 aliases or an unverified NPU architecture

## Expected Paths

- `src/core/diagnostics/system_inventory.cpp`
- `src/core/diagnostics/system_inventory.h`
- `src/core/diagnostics/linux_sysfs.cpp`
- `src/core/diagnostics/linux_sysfs.h`
- `src/core/diagnostics/compatibility.cpp`
- `src/core/diagnostics/compatibility.h`
- `tests/diagnostics/system_inventory_test.cpp`
- `tests/fixtures/diagnostics/sysfs/`
- `docs/CLI.md`

## Definition of Done

- [ ] CPU model/topology, total/available memory, GPU name/architecture/CU count/memory/clocks, NPU identity/resources, Linux kernel, amdgpu/amdxdna drivers, ROCm/HIP, XRT, firmware, clock state, power mode, and available temperatures are represented with source and availability metadata
- [ ] A supported machine reports exact normalized architecture identifiers `gfx1151` and `XDNA2`/`AIE2P`; aliases are retained only as raw evidence
- [ ] Version policy is derived from M001 pinned versions and produces supported, unsupported, or unknown verdicts
- [ ] Unsupported/missing driver, runtime, or firmware errors state detected value, required value/range, evidence path/query, and remediation hint and make diagnose exit nonzero
- [ ] Missing optional telemetry is `unavailable` with reason rather than fabricated zero
- [ ] Fixture tests cover supported Strix Halo, wrong GPU, absent amdxdna, malformed sysfs, permission denial, unsupported versions, and missing optional sensors
- [ ] One focused PR changes inventory collection and compatibility reporting only

## Development Loop

```sh
nix develop -c cmake --preset test
nix develop -c cmake --build --preset test
git add -A
nix build
nix flake check
./result/bin/strix-server diagnose --json --section inventory > /tmp/strix-inventory.json
nix develop -c ctest --preset test --output-on-failure -R 'system_inventory|compatibility'
test "$(nix develop -c bash -lc './result/bin/strix-server diagnose --json --section inventory | python -c "import json,sys; d=json.load(sys.stdin); print(d[\"inventory\"][\"gpu\"][\"architecture\"])"')" = gfx1151
nix develop -c bash -lc './result/bin/strix-server diagnose --json --section inventory | python -c "import json,sys; d=json.load(sys.stdin); assert d[\"inventory\"][\"npu\"][\"architecture\"] in (\"XDNA2\",\"AIE2P\")"'
# Hardware commands must run on the supported Strix Halo host with access to render and accel device nodes.
```

## Output Artifacts

- /tmp/strix-inventory.json (raw local evidence, not committed)
- Sanitized supported and unsupported inventory fixtures under tests/fixtures/diagnostics/sysfs/

## Stop Conditions

- Pinned supported versions from M001 are absent or contradictory
- The host cannot expose required driver/firmware identity through a documented API or readable sysfs query; record the gap rather than guessing
- The test host is not gfx1151 plus XDNA2
- Required device nodes are inaccessible and the error cannot distinguish permissions from missing hardware

## ROADMAP Traceability

- docs/ROADMAP.md Milestone 1 task 2
- docs/ROADMAP.md Milestone 1 exit criteria 1 and 4
- docs/PROJECT_STATUS.md#platform
- docs/PERFORMANCE.md#target
- docs/GPU_BACKEND.md#runtime-components
- docs/NPU_BACKEND.md#backend-objects

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
