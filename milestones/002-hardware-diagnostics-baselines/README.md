# M002: Hardware Diagnostics and Baselines

Roadmap source: [`docs/ROADMAP.md`, Milestone 1](../../docs/ROADMAP.md)

## Objective

Complete ROADMAP Milestone 1.

## Cards

| Card | Dependencies |
| --- | --- |
| [M002-C001: Add the strix-server diagnose command and stable output contract](001-diagnose-command-contract.md) | `M001-C002`, `M001-C003`, `M001-C008` |
| [M002-C002: Collect and validate the Strix Halo hardware and software inventory](002-hardware-software-inventory.md) | `M002-C001`, `M001-C005` |
| [M002-C003: Measure sustained CPU, GPU, and NPU-visible memory bandwidth](003-memory-bandwidth-baselines.md) | `M002-C007`, `M002-C004`, `M002-C005` |
| [M002-C004: Add deterministic HIP allocation, copy, kernel, event, and graph smoke tests](004-hip-deterministic-smoke.md) | `M002-C007`, `M001-C005`, `M001-C008` |
| [M002-C005: Add deterministic XRT discovery, context, buffer, command, and completion smoke tests](005-xrt-deterministic-smoke.md) | `M002-C007`, `M001-C005`, `M001-C008` |
| [M002-C006: Measure first-touch, page-fault, prefault, and thermal steady-state behavior](006-memory-fault-thermal-baselines.md) | `M002-C003`, `M002-C004`, `M002-C005` |
| [M002-C007: Define the canonical machine fingerprint and benchmark artifact binding](007-machine-fingerprint-artifact.md) | `M002-C001`, `M002-C002`, `M001-C005`, `M001-C008` |

## Roadmap Coverage

- tasks: {'roadmapTask': 'M1-T1 Implement strix-server diagnose', 'cards': ['M002-C001']}
- tasks: {'roadmapTask': 'M1-T2 Record CPU, memory, GPU, NPU, driver, runtime, firmware, clock, and power information', 'cards': ['M002-C002']}
- tasks: {'roadmapTask': 'M1-T3 Measure sustained CPU, GPU, and NPU-visible memory bandwidth', 'cards': ['M002-C003']}
- tasks: {'roadmapTask': 'M1-T4 Add HIP allocation, copy, launch, event, and graph smoke tests', 'cards': ['M002-C004']}
- tasks: {'roadmapTask': 'M1-T5 Add XRT device discovery, context, buffer, command, and completion smoke tests', 'cards': ['M002-C005']}
- tasks: {'roadmapTask': 'M1-T6 Measure first-touch, page-fault, prefault, and sustained thermal behavior', 'cards': ['M002-C006']}
- tasks: {'roadmapTask': 'M1-T7 Produce a structured machine fingerprint used by every benchmark artifact', 'cards': ['M002-C007']}
- exitCriteria: {'roadmapExitCriterion': 'M1-E1 Diagnostics identify gfx1151 and XDNA2 correctly', 'cards': ['M002-C002', 'M002-C007']}
- exitCriteria: {'roadmapExitCriterion': 'M1-E2 HIP and XRT can each execute a deterministic device program repeatedly', 'cards': ['M002-C004', 'M002-C005']}
- exitCriteria: {'roadmapExitCriterion': 'M1-E3 Baseline bandwidth and thermal reports are retained', 'cards': ['M002-C003', 'M002-C006', 'M002-C007']}
- exitCriteria: {'roadmapExitCriterion': 'M1-E4 Unsupported driver or firmware combinations fail with actionable errors', 'cards': ['M002-C002', 'M002-C005', 'M002-C007']}

## Milestone Completion Rule

The milestone is complete only when every card's Definition of Done passes and
every ROADMAP exit criterion mapped in these cards has retained test or report
evidence. A skipped required-hardware gate is not a pass.
