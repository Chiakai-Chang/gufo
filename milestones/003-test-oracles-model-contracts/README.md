# M003: Test Oracles and Model Contracts

Roadmap source: [`docs/ROADMAP.md`, Milestone 2](../../docs/ROADMAP.md)

## Objective

Deliver deterministic source inspection, compiled Qwen model contracts, exact text tokenization, independent CPU oracles, teacher-forced full-logit artifacts and comparison, byte-exact candidate SHQ vectors, and the committed capability suite without advancing into accelerator kernels or model serving.

## Cards

| Card | Dependencies |
| --- | --- |
| [M003-C001: Implement the common tensor descriptor and checked arithmetic](001-checked-tensor-descriptor.md) | `M001-C008` |
| [M003-C002: Complete deterministic safetensors index and shard inspection](002-safetensors-inspection-hardening.md) | `M001-C008` |
| [M003-C003: Add the compiled ModelKind registry](003-compiled-model-kind-registry.md) | `M001-C008`, `M003-C001` |
| [M003-C004: Define the Qwen3.5-0.8B explicit model contract](004-qwen35-08b-descriptor.md) | `M003-C001`, `M003-C003` |
| [M003-C005: Reuse the Qwen descriptor family for Qwen3.8-27B text-only production metadata](005-qwen38-27b-text-descriptor-reuse.md) | `M003-C004` |
| [M003-C006: Implement exact Qwen tokenizer conformance](006-qwen-tokenizer-conformance.md) | `M003-C004`, `M003-C005` |
| [M003-C007: Implement the bounded Qwen chat template contract](007-qwen-chat-template-contract.md) | `M003-C006` |
| [M003-C008: Add small high-precision CPU operator oracles](008-high-precision-cpu-operator-oracles.md) | `M003-C001`, `M003-C004` |
| [M003-C009: Harden the full-quality teacher-logit capture workflow and artifact](009-versioned-teacher-logit-capture.md) | `M003-C002`, `M003-C006`, `M003-C007` |
| [M003-C010: Implement validated matched-token full-logit comparison](010-matched-token-logit-comparator.md) | `M003-C009` |
| [M003-C011: Commit byte-exact candidate SHQ4/SHQ6/SHQ8 conformance vectors](011-candidate-shq-byte-vectors.md) | `M003-C001`, `M003-C008` |
| [M003-C012: Commit the capability suite, provenance, and extraction fixtures](012-capability-suite-provenance-fixtures.md) | `M001-C008` |
| [M003-C013: Assemble the Milestone 2 oracle and contract exit gate](013-milestone2-contract-gate.md) | `M003-C002`, `M003-C003`, `M003-C004`, `M003-C005`, `M003-C006`, `M003-C007`, `M003-C008`, `M003-C009`, `M003-C010`, `M003-C011`, `M003-C012` |

## Roadmap Coverage

- tasks: {'M2-task-1-common-tensor-descriptor': ['M003-C001'], 'M2-task-2-safetensors-inspection': ['M003-C002'], 'M2-task-3-ModelKind-registry': ['M003-C003', 'M003-C005'], 'M2-task-4-Qwen-contracts': ['M003-C004', 'M003-C005', 'M003-C007'], 'M2-task-5-tokenizer-chat-template': ['M003-C006', 'M003-C007'], 'M2-task-6-CPU-operator-oracles': ['M003-C008'], 'M2-task-7-teacher-logit-capture': ['M003-C009'], 'M2-task-8-logit-artifact-comparison': ['M003-C009', 'M003-C010'], 'M2-task-9-SHQ-conformance-vectors': ['M003-C011'], 'M2-task-10-capability-suite': ['M003-C012']}
- exits: {'M2-exit-1-deterministic-safetensors-malformed-rejection': ['M003-C001', 'M003-C002', 'M003-C013'], 'M2-exit-2-tokenization-matches-pinned-source': ['M003-C006', 'M003-C007', 'M003-C013'], 'M2-exit-3-byte-exact-CPU-quant-dequant': ['M003-C008', 'M003-C011', 'M003-C013'], 'M2-exit-4-teacher-forced-full-vocabulary-capture-compare': ['M003-C009', 'M003-C010', 'M003-C013'], 'M2-exit-5-source-dtype-distinct-from-accumulation-dtype': ['M003-C008', 'M003-C009', 'M003-C013']}

## Milestone Completion Rule

The milestone is complete only when every card's Definition of Done passes and
every ROADMAP exit criterion mapped in these cards has retained test or report
evidence. A skipped required-hardware gate is not a pass.
