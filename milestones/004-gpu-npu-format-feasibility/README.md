# M004: Early GPU and NPU Format Feasibility

Roadmap source: [`docs/ROADMAP.md`, Milestone 3](../../docs/ROADMAP.md)

## Objective

Measure the candidate SHQ layouts on gfx1151 and XDNA2, close an independent
blocking GPU-format gate for Milestone 4, and retain non-blocking NPU and
shared-allocation evidence. Unsupported or unstable NPU outcomes are valid
completed research results and never block the GPU MVP. SHQ remains a candidate
contract unless every freeze gate in `docs/PROJECT_STATUS.md` is independently
satisfied; this milestone does not force v1.

## Tracks

- **GPU blocking:** M004-C001–C004 and M004-C011.
- **NPU non-blocking:** M004-C005–C008 and M004-C010.
- **Shared-allocation non-blocking:** M004-C009, M004-C012, and M004-C010.

## Cards

| Card | Dependencies |
| --- | --- |
| [M004-C001: SHQ8 decode GEMV](001-qwen-gfx1151-shq8-decode-gemv.md) | M001-C008, M002-C007, M003-C008, M003-C011, M003-C013 |
| [M004-C002: SHQ4 decode GEMV](002-qwen-gfx1151-shq4-decode-gemv.md) | M001-C008, M002-C007, M003-C008, M003-C011, M003-C013 |
| [M004-C003: SHQ8 small-row/layout study](003-qwen-gfx1151-shq8-small-row-layout-study.md) | M002-C007, M004-C001 |
| [M004-C004: SHQ4 small-row/layout study](004-qwen-gfx1151-shq4-small-row-layout-study.md) | M002-C007, M004-C002 |
| [M004-C005: AIE2P/XRT lifecycle](005-qwen-aie2p-xrt-program-lifecycle.md) | M001-C008, M002-C005, M002-C007, M003-C011, M003-C013 |
| [M004-C006: AIE W8A8/SHQ8](006-qwen-aie2p-w8a8-shq8-matrix.md) | M003-C008, M003-C011, M004-C005 |
| [M004-C007: AIE W4A8/SHQ4](007-qwen-aie2p-w4a8-shq4-matrix.md) | M003-C008, M003-C011, M004-C005 |
| [M004-C008: AIE BF16 and shape buckets](008-qwen-aie2p-bf16-shape-buckets.md) | M003-C008, M004-C005 |
| [M004-C009: Basic XRT BO import into HIP](009-xrt-bo-hip-explicit-handoff-probe.md) | M002-C004, M002-C005, M002-C007, M004-C005 |
| [M004-C010: Sustained GPU/NPU concurrency](010-m3-concurrency-tier-and-format-exit.md) | M004-C005, M004-C006, M004-C007, M004-C008, M004-C012 |
| [M004-C011: Blocking GPU-format gate](011-gpu-format-feasibility-gate.md) | M004-C001, M004-C002, M004-C003, M004-C004 |
| [M004-C012: Handoff stress and tier selection](012-xrt-hip-handoff-stress-and-tier.md) | M004-C003, M004-C004, M004-C009 |

## ROADMAP Task Coverage

| ROADMAP requirement | Cards |
| --- | --- |
| GPU 1: SHQ4 decode GEMV | M004-C002 |
| GPU 2: SHQ8 decode GEMV | M004-C001 |
| GPU 3: Q4/Q8 small-row and prefill candidates | M004-C003, M004-C004 |
| GPU 4: common T16 versus lossless GPU-native view | M004-C003, M004-C004 |
| GPU 5: ISA, occupancy, bandwidth, alignment, tails | M004-C001–C004, M004-C011 |
| NPU 1: minimal model-private AIE2P program through XRT | M004-C005 |
| NPU 2: W4A8, W8A8, and BF16 matrix benchmarks | M004-C006–C008 |
| NPU 3: GEMM versus batch-1 GEMV | M004-C006–C008 |
| NPU 4: T16 ordering and scale epilogues | M004-C006, M004-C007 |
| NPU 5: reusable shape buckets and reconfiguration cost | M004-C006–C008 |
| NPU 6: context, load, submit, DMA, completion overhead | M004-C005–C008 |
| Shared 1: XRT BO/dma-buf import into HIP | M004-C009 |
| Shared 2–5: explicit ordering, stress, transition latency, immutable reads | M004-C012 |
| Shared 6: sustained ROCm+XDNA2 load | M004-C010 |
| Shared 7–8: fallback views/copies and tier decision | M004-C012 |

## Exit-Criterion Coverage

| Exit criterion | Evidence |
| --- | --- |
| CPU/HIP/AIE numerical agreement | M004-C001, C002, C006, C007, C008 |
| Common layout measured on both accelerators | M004-C003, C004, C006, C007, C008 |
| Interoperability tier selected from evidence | M004-C009, C012 |
| SHQ frozen or revised before portable publication | M004-C010 records **candidate/defer** unless every `docs/PROJECT_STATUS.md` gate exists; this milestone does not force v1 |
| Unstable/slower NPU does not block GPU | M004-C010 supported-negative closure; M004-C011 independent GPU gate |

## Completion Rules

- **GPU prerequisite for Milestone 4:** only M004-C011 must pass.
- **NPU/shared research:** cards complete with either supported-positive evidence
  or a retained supported-negative/disabled result. They may remain outside the
  GPU critical path when required hardware is unavailable.
- No card may declare SHQ v1 merely from GPU or import success.
