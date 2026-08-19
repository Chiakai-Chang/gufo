# ROCm/HIP GPU Backend

Status: design draft, 2026-08-11

## Purpose

The GPU backend is a native HIP implementation compiled specifically for the
Strix Halo `gfx1151` RDNA 3.5 GPU.

It must not depend on:

- CUDA headers or libraries.
- `.cu` source files.
- hipify-generated sources.
- PyTorch, ATen, or Triton.
- vLLM runtime components.

hipBLASLt provides selected production BF16 prefill algorithms, while rocBLAS
provides baselines and fallback algorithms. Model-critical paths may still use
custom HIP kernels where profiling demonstrates a benefit. Composable Kernel
provides a fused causal GQA fallback for shapes not handled by a native tile.

## Build

Use CMake with C++20 and HIP as first-class languages:

```cmake
project(strix LANGUAGES CXX HIP)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_HIP_ARCHITECTURES gfx1151)
```

GPU source files use `.hip`. Release packages contain only `gfx1151` code
objects unless a development build explicitly enables another architecture.

Compiler version, flags, ROCm version, kernel source hash, and architecture are
part of every cached artifact key.

## Backend Boundary

The model runtime calls whole operations through raw-pointer interfaces:

```cpp
struct HipExecutionContext {
    hipStream_t stream;
    ScratchArena scratch;
    ProfilingSink* profiling;
};

Status model_prefill(..., HipExecutionContext&);
Status model_decode_batch(..., HipExecutionContext&);
Status model_verify(..., HipExecutionContext&);
```

Kernel entry points use:

- Raw pointers.
- Fixed-width integer dimensions.
- POD configuration structs.
- Explicit stream arguments.
- Preallocated scratch.

Do not pass framework tensor objects into the backend.

## Runtime Components

### Device

- Require `gfx1151`.
- Record CU count, wave mode, available memory, clock state, and runtime version.
- Reject accidental `gfx1100` aliases in production packages.

### Allocator

- Suballocate large aligned arenas.
- Separate immutable weights, persistent session state, KV pages, graph-stable
  buffers, and transient scratch.
- Perform no timed-path allocation.
- Integrate with the shared GPU/NPU allocation broker where visibility is
  proven.

### Streams and Events

Initial stream classes:

- Latency-critical decode.
- Bulk prefill/verification.
- Background copy and persistence.

Additional streams require profile evidence. Shared memory bandwidth and queue
contention can make nominal overlap slower.

### HIP Graphs

Capture stable shape buckets after:

- All allocations are complete.
- Pointer identities are stable.
- Kernel selection is finalized.
- Correctness matches eager execution.

Cache graphs by model, route, physical row width, context policy, and
speculative topology. Graph replay is an overhead optimization, not a memory
bandwidth optimization.

## Kernel Families

### Single-token decode

- Wave32 GEMV.
- Fused SHQ4-T16 unpack, zero correction, scale, and dot.
- Multi-output and dual-projection variants where input reuse is material.
- Fused model-specific tails only when they reduce full wall time.

This path is primarily optimized for:

- Coalesced weight reads.
- Sufficient outstanding memory operations.
- Minimal redundant activation reads.
- Occupancy without excessive registers.
- Low launch count.

### Prefill and batched verification

- 16 x 16 x 16 WMMA-class kernels.
- Compact active-row schedules.
- T16 weights staged into device-friendly fragments.
- Separate algorithms for large dense tensors and small verifier shapes.
- hipBLASLt baseline for large standard GEMMs.

Do not route all `M > 1` shapes through WMMA. Small row counts may still favor
row-tiled GEMV.

### Attention

- Fused RoPE and Q/K preparation where model-specific fusion is beneficial.
- Native Qwen3.8 causal GQA tiles adapted from llama.cpp's MIT online-softmax
  scheduling for long prompts, with Composable Kernel as a shape fallback.
- Chunked prefill keeps activation batches bounded while carrying absolute
  positions and persistent KV/recurrent state across context frontiers.
- Single-token Qwen3.8 attention uses FP32 online softmax without a
  context-sized LDS score array, including at 16K and deeper positions.
- Paged KV attention.
- Separate single-token and batched attention kernels.
- GQA-aware K/V reuse.
- Long-context kernels selected by measured context ranges.

### MoE

- GPU router and top-k baseline.
- Row/expert compaction.
- Grouped expert GEMM for populated expert tiles.
- Selected-expert GEMV for sparse decode.
- Fused weighted expert combination where correctness permits.

### Sampling

- Device-side greedy argmax.
- Temperature, top-k, top-p, penalties, and seeded RNG.
- Grammar and structured-output constraints through a bounded device/host
  contract.
- No full-vocabulary host transfer in the ordinary path.

## Quantization Paths

The shared artifact family is `SHQ-T16`. Its first required tensor encodings
are `SHQ4-T16`, `SHQ8-T16`, and BF16.

GPU decode normally consumes BF16 activations directly. Prefill and verifier
paths may quantize activations to INT8 once and reuse that buffer across
multiple projections.

Required quant kernels:

- BF16/F16 to dynamic INT8 activation blocks.
- SHQ4-T16 GEMV.
- SHQ4-T16 small-row GEMM.
- SHQ4-T16 WMMA staging and GEMM.
- SHQ8-T16 and BF16 sensitive-tensor paths.

Each model supplies its own implementation of the required paths. This list is
an interface requirement, not a shared kernel library.

All backend paths must match one CPU dequantization contract before full-model
testing.

## gfx1151-Specific Rules

- Assume wave32 only in kernels explicitly compiled and tested for wave32.
- Use native `gfx1151` compilation and inspect generated ISA for critical paths.
- Do not transfer launch geometry from W7900 or `gfx1100` without a new
  benchmark.
- Treat approximately 200-221 GB/s as the practical read ceiling until the
  exact HIP workload measures otherwise.
- Avoid repeated-lane loads that have shown unsafe behavior in generic
  gfx1151 Triton patterns.
- Keep resource usage low enough to maintain useful occupancy on 40 CUs.
- Tune thermal and sustained-clock behavior, not only short benchmark peaks.

## Model Isolation

Each compiled-in model implementation owns its GPU kernels:

```text
models/<model>/gpu/gfx1151/
```

Each model is compiled as an independent object target with unique host and
device symbol prefixes. The final link places all targets in the single
`strix-server` executable, but does not merge their kernel ownership.

There is no global numerical kernel registry, shared tuning table, or
production `common/kernels` directory. A model binds its tensors to its own
dispatch table during load.

Shared runtime helpers may provide allocation, graph capture, raw launch
plumbing, streams, events, error handling, and profiling. They do not contain:

- Dot products, dequantization, attention, MoE, sampling, or model epilogues.
- Kernel launch geometry or tuning constants.
- Model tensor-shape assumptions.
- Generated HIP kernel bodies.

When a kernel idea is useful to another model, copy its source into the other
model and establish independent tests and ownership. Do not use symlinks,
source includes across model directories, or a generator output shared by
multiple model targets.

Different size variants of the same model reuse that model's private kernel
set. Their manifests bind tensors to model-local encoding specializations.

A model-private change must preserve the compiled code-object and ISA hashes of
other models. A shared-runtime change triggers independent correctness and
performance gates for every affected model.

## Tests

- Kernel CPU oracle tests.
- Eager versus HIP graph tests.
- Exact greedy token tests.
- Full-logit and layer-output comparisons.
- Alignment, tails, and non-multiple dimensions.
- Stream and event ordering.
- Cancellation at scheduler commit boundaries.
- No allocation during timed execution.
- Scratch and graph lifetime accounting.
- Long-duration thermal benchmarks.
- rocprof kernel and memory traces for every promoted route.
- Direct runtime versus server-path parity.
- Model-private dependency-graph enforcement.
- Unchanged code-object and ISA hashes for untouched models.
