# M001: Milestone 0: Repository and Toolchain

Roadmap source: [`docs/ROADMAP.md`, Milestone 0](../../docs/ROADMAP.md)

## Objective

Complete ROADMAP Milestone 0.

## Cards

| Card | Dependencies |
| --- | --- |
| [M001-C001: Normalize the repository and MIT license baseline](001-repository-license-baseline.md) | None |
| [M001-C002: Add project NOTICE and initial third-party inventory](002-initial-third-party-notices.md) | `M001-C001` |
| [M001-C003: Create the C++20 source layout and placeholder strix-server](003-cpp20-server-skeleton.md) | `M001-C001` |
| [M001-C004: Add development, release, sanitizer, and test presets](004-cmake-build-presets.md) | `M001-C003` |
| [M001-C005: Pin and publish the supported Strix Halo toolchain matrix](005-supported-toolchain-pins.md) | `M001-C001` |
| [M001-C006: Reject CUDA at source, link, symbol, and runtime-closure boundaries](006-reject-cuda-contamination.md) | `M001-C003`, `M001-C005` |
| [M001-C007: Add formatting, static-analysis, dependency, and documentation checks](007-repository-quality-checks.md) | `M001-C002`, `M001-C003`, `M001-C004`, `M001-C005`, `M001-C006` |
| [M001-C008: Create and document the initial PR test command](008-single-pr-test-entrypoint.md) | `M001-C004`, `M001-C006`, `M001-C007` |

## Roadmap Coverage

- taskMapping: {'M0 task 1': ['M001-C001'], 'M0 task 2': ['M001-C002'], 'M0 task 3': ['M001-C003'], 'M0 task 4': ['M001-C004'], 'M0 task 5': ['M001-C005'], 'M0 task 6': ['M001-C006'], 'M0 task 7': ['M001-C007'], 'M0 task 8': ['M001-C008']}
- exitCriterionMapping: {'A clean checkout configures and builds on the supported Strix Halo Linux environment': ['M001-C005', 'M001-C008'], 'A placeholder strix-server --version runs': ['M001-C003', 'M001-C008'], 'The binary and dependency scan contain no CUDA dependency': ['M001-C006', 'M001-C007', 'M001-C008'], 'The PR test command succeeds from one documented entry point': ['M001-C008']}
- allRoadmapTasksMapped: True
- allExitCriteriaMapped: True
- cardCount: 8

## Milestone Completion Rule

The milestone is complete only when every card's Definition of Done passes and
every ROADMAP exit criterion mapped in these cards has retained test or report
evidence. A skipped required-hardware gate is not a pass.
