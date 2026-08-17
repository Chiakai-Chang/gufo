# Strix Halo Performance Engineering

Status: design draft, 2026-08-11

## Purpose

This document defines how performance-critical host code, HIP kernels, AIE2P
programs, memory layouts, and GPU/NPU routes are designed and reviewed for
Strix Halo.

The project optimizes end-to-end request behavior, not isolated peak FLOPS or
TOPS. A kernel is promoted only when it preserves its numerical contract and
improves the declared serving workload.

## Target

The only production target is:

```text
OS:          Linux x86-64
APU:         AMD Strix Halo
GPU:         gfx1151, RDNA 3.5, up to 40 CUs
GPU waves:   wave32
NPU:         XDNA2/AIE2P, 4 rows x 8 columns
NPU L2:      4 MiB software-managed memory-tile capacity
Memory:      unified LPDDR5X, up to 128 GiB
```

Runtime probing records the actual CU count, clocks, memory configuration,
available AIE columns, driver and firmware versions, thermals, and power mode.
Do not assume that every machine exposes the top configuration.

## Optimization Order

For every workload:

1. Define the numerical and state contract.
2. Measure end-to-end wall time and identify the dominant phase.
3. Classify the phase as bandwidth, compute, launch, synchronization, or
   scheduling limited.
4. Build a narrow reproducer and CPU or canonical oracle.
5. Optimize the dominant cost.
6. Re-run full correctness and request-level performance.
7. Retain the result only when the complete route improves.

Do not begin with a device instruction simply because it has a high advertised
throughput.

## Roofline Model

Estimate arithmetic intensity:

```text
intensity = useful operations / external-memory bytes
```

Then compare against measured, not theoretical, machine ceilings.

### Single-token decode

Dense and selected-expert GEMV usually read each active weight once for one
token. Decode performance is primarily determined by:

- Compressed bytes read per active parameter.
- Effective LPDDR bandwidth.
- Coalescing and outstanding memory operations.
- Weight and metadata duplication.
- Launch count.
- KV traffic at the active context length.

Peak matrix throughput is secondary when `M=1`.

### Prefill and verification

Larger row counts reuse weights and increase arithmetic intensity. Performance
depends more strongly on:

- Matrix-instruction utilization.
- Tile reuse.
- Activation packing.
- NPU memory-tile and GPU LDS reuse.
- Padding and shape buckets.
- DMA and synchronization overlap.

### Measured Sustained Memory Bandwidth Baselines

Sustained bandwidth across CPU, GPU (`gfx1151`), and NPU (`XDNA2`) buffer paths on AMD Strix Halo (128 GiB unified LPDDR5X, fingerprint `bb565d5eff3a9f23b4ac3f1ff03f66bebef651e57093cd558c4cded823358849`):

| Backend | Path | Allocation Type | Working Set | Median (GB/s) | P95 (GB/s) | Status |
| --- | --- | --- | --- | --- | --- | --- |
| CPU | `cpu_copy` | `host_pageable` | 256 MiB | 50.62 | 51.11 | `completed` |
| CPU | `cpu_read` | `host_pageable` | 256 MiB | 13.88 | 13.89 | `completed` |
| CPU | `cpu_write` | `host_pageable` | 256 MiB | 37.96 | 38.37 | `completed` |
| HIP | `hip_h2d` | `hip_device_memory` | 256 MiB | 68.36 | 68.56 | `completed` |
| HIP | `hip_device_copy` | `hip_device_memory` | 256 MiB | 211.03 | 211.63 | `completed` |
| HIP | `hip_d2h` | `hip_device_memory` | 256 MiB | 55.89 | 56.42 | `completed` |
| XRT | `xrt_bo_sync_to_device` | `xrt_bo_dma_sync` | 64 MiB | 127.06 | 127.75 | `completed` |
| XRT | `xrt_bo_sync_from_device` | `xrt_bo_dma_sync` | 64 MiB | 120.35 | 127.74 | `completed` |

These measured ceilings, rather than theoretical LPDDR bandwidth, govern decode GEMV and weight-streaming roofline bounds.

## Measurement Rules

Every retained performance result records:

- Exact server revision and build type.
- Model, weight artifact hash, quantization, and KV profile.
- Prompt token IDs and output-token count.
- Concurrency and service class.
- GPU/NPU route and physical shape.
- Kernel, graph, AIE program, and compiler artifact hashes.
- Linux kernel, `amdxdna`, firmware, ROCm, HIP compiler, and AIE compiler.
- Memory configuration, power mode, clocks, and temperature.
- Warmup policy and every measured repetition.
- Correctness artifact proving the tested route is valid.

Compare baseline and candidate on the same machine and in alternating order.
Report median and tail latency, not the best run.

## C++ Host Code

### Ownership and allocation

- Use RAII for every file, mapping, queue, event, graph, context, and
  allocation.
- Allocate model, KV, graph, and scratch resources before readiness.
- Perform no general heap allocation in decode dispatch.
- Reuse bounded vectors, command records, and response buffers.
- Keep scheduler-visible ownership explicit.
- Reject integer overflow before calculating tensor or allocation bytes.

### Threading

Use separate bounded execution domains for:

- HTTP and SSE I/O.
- Tokenization and detokenization.
- The single-owner scheduler.
- GPU submission and completion.
- NPU submission and completion.
- Background storage I/O.

Do not create one thread per request. Avoid oversubscribing CPU cores merely to
hide device latency.

Thread affinity, priority, and busy polling are promoted only after measuring
system-wide effects. Latency-critical polling must yield or sleep when the
server is idle.

### Synchronization

- Prefer single-owner state machines and message passing.
- Do not hold a process-wide mutex while waiting for a device.
- Keep device completion callbacks small.
- Batch page-table and row-map publication.
- Use generation-tagged handles to reject stale completions.
- Avoid CPU round trips between kernels in one graph or AIE program.

### Data structures

- Keep hot scheduler arrays compact and contiguous.
- Use structure-of-arrays layouts for fields consumed independently.
- Separate cold diagnostics and strings from hot request state.
- Use fixed-width integer types in serialized and device-visible structures.
- Align shared command and tensor metadata explicitly.
- Measure cache behavior before adding pointer-rich abstractions.

## HIP Kernel Guidance

### General rules

- Compile production kernels directly for `gfx1151`.
- Treat wave32 as an explicit kernel contract.
- Inspect generated ISA for critical kernels.
- Use aligned vector loads only when the alignment contract proves them legal.
- Provide bounded tail paths for every non-multiple dimension.
- Keep register and LDS use low enough to preserve useful occupancy.
- Benchmark sustained clocks, not only short bursts.
- Never inherit launch geometry from `gfx1100`, W7900, or CUDA without a new
  gfx1151 measurement.

### Decode GEMV

Optimize in this order:

1. Read each weight and scale as few times as possible.
2. Make adjacent lanes read adjacent packed bytes.
3. Reuse one activation across multiple output accumulators when registers
   permit.
4. Fuse SHQ unpack, zero correction, scaling, and dot product.
5. Fuse adjacent projections only when total bytes and launches decrease.
6. Reduce partial sums with wave operations before using LDS.
7. Keep the output epilogue bounded and branch-free.

Candidate shapes include:

- One wave per output tile.
- Multiple output channels per wave.
- Multiple waves per workgroup when activation or metadata reuse justifies LDS.

The chosen shape is tensor-specific. Large hidden sizes, small projections,
MoE expert matrices, and output heads need separate measurements.

Avoid:

- Dequantizing full weight tiles into global memory.
- Reading scale or zero-point metadata once per weight.
- One kernel launch per small tensor transformation.
- Excessive output accumulators that spill registers.
- Repeated-lane load patterns that are not independently validated on gfx1151.

### Prefill and small-batch GEMM

- Maintain separate paths for small rows and large matrices.
- Use row-tiled GEMV when WMMA setup and padding dominate.
- Use WMMA-class tiles when row count and dimensions provide reuse.
- Stage only data that will be reused enough to repay LDS traffic.
- Fuse activation quantization when the packed result feeds several
  projections.
- Retain hipBLASLt or rocBLAS as a correctness and performance baseline for
  supported standard shapes.

Do not route every `M > 1` operation through one matrix kernel.

### Attention and KV

- Use distinct decode and prefill attention kernels.
- Read only live KV spans.
- Fuse scale application for quantized KV into the consuming kernel.
- Reuse GQA/MQA K/V across query heads.
- Select context-range-specific kernels.
- Keep page-table entries compact and device resident.
- Avoid rebuilding host pointer arrays for every token.

Long-context performance reports include both attention wall time and complete
decode wall time.

### MoE

- Keep routing on device where practical.
- Compact rows by expert before grouped GEMM.
- Use selected-expert GEMV for sparse decode.
- Use grouped GEMM only after population exceeds its measured crossover.
- Avoid loading inactive expert weights.
- Fuse weighted combination when it removes traffic without changing routing
  or accumulation semantics.

### Sampling

- Keep ordinary full-vocabulary logits on device.
- Use deterministic request-owned RNG state.
- Separate greedy, top-k, top-p, and constrained paths when one generic kernel
  adds material overhead.
- Transfer only the selected token and bounded diagnostics to the host.

## HIP Graphs

Capture graphs only after:

- Allocations and pointer identities are stable.
- The physical shape is known.
- Kernel selection has been promoted.
- Eager execution passes correctness.

Graphs are keyed by all state that changes topology, including model kind,
route, physical width, context class, KV profile, and speculative shape.

Measure graph replay end to end. Reducing launches does not guarantee a win when
the phase is bandwidth-bound.

## AIE2P Program Guidance

### Workload shape

The NPU is best used for stable, sufficiently large row batches. Use fixed
program families for row buckets and requested column counts.

- Prefer native INT8 x INT4/UINT4 operations for SHQ4-T16 tensors.
- Match the native `4 x 16 x 16` mixed-precision microtile.
- Avoid padding a tiny logical batch to a large physical program.
- Compile smaller-column programs for narrow work and shared-resource
  conditions.
- Key performance records by columns actually assigned by `amdxdna`.

### Memory movement

Each AIE column has DMA resources between host DDR and its memory tile. The
software-managed memory-tile capacity is limited, so program design must
explicitly schedule:

- Input activation DMA.
- Packed weight DMA.
- Scale and zero-point metadata.
- Double-buffered tile execution.
- Output or partial-result DMA.

The program should overlap DMA and compute when tile sizes permit. Count all
host-memory bytes in the route roofline.

Do not unpack the complete model or tensor into INT8/BF16 host memory. Expand
or reinterpret packed values at the smallest tile scope supported by the
microkernel.

### Programs and contexts

- Compile overlays and `ctrlcode` ahead of time.
- Keep program metadata immutable.
- Reuse workload contexts when supported and healthy.
- Account for driver-managed instruction and command buffers.
- Minimize context creation and overlay reconfiguration in request paths.
- Recreate contexts and reload programs after NPU suspend, reset, or firmware
  restart.

### NPU epilogues

Fuse where practical:

- U4 zero-point correction.
- Activation and weight scales.
- Bias.
- Output conversion.
- Simple model-specific activation.

An epilogue is retained only when it reduces host-memory traffic or
synchronization without increasing numerical error beyond the route contract.

## GPU and NPU Together

### Shared bandwidth

GPU and NPU external-memory bandwidth does not add. Concurrent work can improve
performance only when:

- The workload has enough arithmetic intensity.
- Each device reads disjoint weight regions or works on independent requests.
- DMA and compute overlap exceeds synchronization overhead.
- Protected GPU decode latency remains within budget.

Do not split single-token dense decode merely to occupy the NPU.

### Preferred concurrency

Initial priorities:

```text
GPU: interactive decode
NPU: batched prefill, proposal, or verification
CPU: tokenization, transport, and bounded orchestration
```

For one large prefill or verification operator, test `SPLIT_N`,
`SPLIT_K_REDUCE`, and single-device baselines. For MoE, test expert ownership
before splitting individual expert matrices.

### Synchronization

- Submit complete operators or stages before crossing devices.
- Use release/acquire completion tokens.
- Avoid a CPU wait between every layer.
- Keep reductions on the consuming device where possible.
- Include activation conversion, barriers, and copies in reported wall time.

## Quantization Performance

Quality is evaluated independently from speed. For each quantization candidate
measure:

- Effective bytes per weight including scales, zero points, padding, and index
  metadata.
- GPU decode bandwidth efficiency.
- GPU prefill wall time.
- NPU DMA plus compute wall time.
- Shared-layout penalty versus an ideal backend-native layout.
- Cost and memory of optional lossless repacks.
- Full-model quality at matched size and matched performance.

A nominally smaller format is rejected when unpack, metadata, or restoration
cost makes the complete model slower.

## Profiling

Use:

- `rocprofv3` or current ROCm profiling interfaces for HIP timelines and
  counters.
- Generated ISA inspection for critical gfx1151 kernels.
- `amdxdna` DRM client usage statistics and available driver telemetry.
- Explicit scheduler spans for queueing, submission, synchronization, and
  commit.
- CPU sampling or tracing for tokenizer, JSON, and scheduler overhead.

Profiler collection must not run inside the default production path. Markers
and bounded counters remain available in release builds.

## Promotion Gates

A performance change is promoted only when:

- The targeted numerical and state tests pass.
- The released model remains within its declared quality budget.
- The end-to-end metric improves beyond measured noise.
- Single-request latency does not regress outside its declared budget.
- Memory use and pressure behavior remain acceptable.
- Sustained performance survives thermal steady state.
- Multi-request fairness remains within policy.
- The exact benchmark artifact is retained.

Hard-coded prompt, token, shape, or request-ID special cases are prohibited.

## Review Checklist

- What end-to-end phase is dominant?
- Is the phase bandwidth, compute, launch, or synchronization limited?
- What external bytes and useful operations are expected?
- Which exact shape and route are optimized?
- What CPU or canonical oracle validates it?
- Are allocations, first touch, and JIT work outside the timed path?
- Does the candidate work for tails and non-multiple dimensions?
- What are register, LDS, AIE memory-tile, and context costs?
- Does it affect GPU/NPU shared bandwidth?
- Does single-request latency remain protected?
- Is the result reproducible from a structured artifact?
