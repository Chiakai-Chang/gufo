# Optimization and Research Ideas

This document collects performance, runtime, kernel, and algorithmic ideas for
Strix-Halo.cpp. It is a research backlog, not a claim that every idea will be
beneficial. Every optimization must be measured on the supported production
target: Linux x86-64 on AMD Strix Halo (`gfx1151` GPU and XDNA2 NPU).

## Goals

- Increase prompt-processing throughput (`pp`).
- Increase token-generation throughput (`tg`).
- Reduce time to first token and inter-token latency.
- Improve long-context performance without breaking cache correctness.
- Exploit the complete Strix Halo device: CPU, `gfx1151`, unified memory, and
  XDNA2.
- Preserve exact model semantics unless an approximation is explicitly
  documented, measured, and selectable.
- Keep optimization work modular enough for independent agents to implement and
  review safely.

## Rules for Optimization Work

1. Capture a correctness and performance baseline before changing a kernel.
2. Change one major variable at a time: arithmetic, launch policy, layout, or
   dispatch threshold.
3. Keep a reference implementation available.
4. Report the selected backend and why it was selected.
5. Benchmark warm and cold execution separately.
6. Record model, context, batch, clock/power mode, ROCm version, kernel, and
   firmware with every result.
7. Treat speedups measured on CDNA, CUDA, or discrete RDNA GPUs as hypotheses,
   not expected Strix Halo results.
8. Run the canonical Nix gate and gfx1151-specific correctness tests before
   accepting an optimization.

## Immediate Correctness Prerequisite

### Baseline-to-tiled FP16 KV-cache transition

Incremental prefill can start with baseline attention and later cross the
optimized-attention threshold. The baseline path updates the FP32 KV cache,
while tiled attention consumes a separate FP16 cache and currently packs only
the active chunk. Switching to tiled attention with a nonzero start position may
therefore read an FP16 prefix that was never populated.

Before extending tiled or speculative verification paths, choose and validate
one of these policies:

- maintain FP16 cache coherence from the first chunk;
- convert the existing FP32 prefix exactly once when transitioning;
- track cache-valid ranges and populate missing ranges;
- prohibit a nonzero-position tiled transition and remain on baseline;
- use one cache representation for all relevant attention backends.

Required tests include baseline-only, tiled-only, and baseline-to-tiled
transitions around the dispatch threshold.

## Measurement Infrastructure

### Kernel benchmark harness

Add focused benchmarks for:

- GEMV by `M`, `K`, type, and weight layout;
- GEMM by prompt batch and projection shape;
- single-token attention by context length;
- batched attention by prompt length and start position;
- DeltaNet recurrence by batch and state size;
- conversion, cache packing, RoPE, normalization, and residual kernels;
- graph launch versus ordinary launch;
- draft and verification stages for speculative decoding.

Suggested context matrix:

```text
128, 512, 1024, 2048, 4096, 8192, 16384, 32768
```

Suggested reporting:

```text
median, p10, p90, tokens/s, effective GB/s, occupancy,
VGPR usage, LDS usage, launches/token, accepted tokens/draft
```

### Dispatch telemetry

In diagnostic or benchmark mode, report:

- selected attention backend;
- selected GEMV/GEMM strategy;
- rejected fast paths and their support predicate;
- hipBLASLt algorithm identifier;
- graph cache hit/miss;
- cache representation and valid range;
- speculative draft length and acceptance rate.

Production logging should remain disabled or sampled to avoid perturbing
latency.

### Stable benchmark records

Store machine-readable results keyed by:

- hardware fingerprint;
- model and tensor layout;
- commit;
- ROCm/HRX/XRT versions;
- kernel and firmware;
- power profile;
- benchmark case.

Normal PR gates should enforce correctness and compilation. Performance
regression thresholds should run on a dedicated, stable Strix Halo machine.

## GPU Kernel Ideas

### 1. gfx1151-specific GEMV autotuning

Decode is often limited by model-weight bandwidth. Candidate strategies:

- wave32-specialized kernels;
- multiple output rows per workgroup;
- register reuse of the activation vector;
- vectorized/coalesced weight loads;
- input-size-specific workgroup configurations;
- prepacked or transposed device-native weights;
- fused conversion, bias, gate, or activation epilogues;
- persistent algorithm choices keyed by tensor shape and hardware.

Keep selection independent from implementation:

```cpp
enum class GemvStrategy {
  kBaseline,
  kWave32,
  kWideInput,
  kQuantized,
};
```

Do not assume a single GEMV kernel is optimal for embeddings, attention
projections, FFN projections, and logits.

### 2. hipBLASLt plan persistence and shape tuning

The engine already caches hipBLASLt plans during execution. Extend this with:

- offline tuning on the target machine;
- a persisted shape-to-algorithm database;
- separate selections for cold and steady-state execution;
- workspace-aware selections;
- exact fallback to the current path when a stored algorithm is unavailable;
- regression detection when ROCm changes algorithm behavior.

### 3. Flash-Decoding / split-K decode attention

Single-token decode has one query and an increasingly long KV sequence. At long
contexts, one workgroup per head may not expose enough parallelism.

Candidate design:

1. Split the KV sequence across multiple workgroups.
2. Compute partial online-softmax maxima, sums, and output vectors.
3. Combine partial results with a small stable reduction.
4. Avoid materializing the complete attention-score vector.

Possible dispatch:

```text
short context     -> existing online-softmax kernel
medium context    -> wider/vectorized kernel
long context      -> split-K Flash-Decoding
very long context -> split-K plus compressed or paged KV
```

Numerical tests must compare logits and attention outputs across context
boundaries and split counts.

### 4. Improved tiled prefill attention

The native Qwen tiled kernel can be explored along these dimensions:

- query rows per workgroup;
- KV tile width;
- heads or head pairs per block;
- wave32 work partitioning;
- LDS bank-conflict avoidance;
- FP16/BF16 load format with FP32 accumulation;
- vectorized cache packing;
- online-softmax reduction layout;
- fused Q/K normalization, RoPE, and cache write;
- start-position-aware cache reuse;
- tile selection by prompt shape rather than one fixed threshold.

Keep baseline, tiled, CK, and GEMM implementations independently callable for
correctness and benchmarking.

### 5. Targeted kernel fusion

Candidates that may eliminate meaningful global-memory traffic:

- RMSNorm plus projection preparation;
- Q/K normalization plus RoPE;
- RoPE plus KV-cache write;
- attention gate plus output conversion;
- residual add plus the next RMSNorm;
- FFN projection plus SwiGLU;
- SSM post-normalization, gate, and residual.

Avoid whole-layer mega-kernels until profiling proves they are beneficial.
Excessive fusion can increase VGPR pressure, reduce occupancy, increase compile
time, and make shape fallback difficult.

### 6. Persistent or cooperative decode execution

Research whether several small decode operations can remain in a persistent
kernel or cooperative execution loop. Potential benefits are fewer launches and
more on-chip state reuse. Risks include global synchronization, poor occupancy,
register pressure, and difficult fallback behavior.

Treat this as a later experiment after HIP Graphs quantify the launch-overhead
ceiling.

## Runtime and Scheduling Ideas

### 7. HIP Graph capture

Decode repeatedly runs a stable launch sequence with stable allocations. Explore:

- complete-token graph capture;
- per-layer graph capture when full capture is unsupported;
- graph caching by model, execution route, and shape bucket;
- patching position-dependent parameters;
- graceful fallback to ordinary launches;
- measuring CPU launch gaps before and after capture.

### 8. Continuous batching

For server throughput, separate prompt and decode scheduling and dynamically
batch compatible requests. Required work includes:

- per-session state ownership;
- KV allocation policy;
- fairness and latency objectives;
- cancellation and backpressure;
- shape-aware batching;
- deterministic correctness tests.

This may not benefit single-user latency and should remain selectable.

### 9. Paged KV cache and prefix sharing

Paged KV can support continuous batching, prefix reuse, and reduced contiguous
allocation pressure. It can also add address calculation and metadata overhead.
Prioritize it when multi-session serving or long-context allocation becomes a
measured bottleneck.

Prefix caching should include model, tokenizer, chat template, precision,
position policy, and backend compatibility in its key.

### 10. Unified-memory-aware placement

Strix Halo shares physical memory across CPU, GPU, and NPU. Explore:

- weight mapping versus explicit device allocation;
- page pinning and migration behavior;
- prefetching before layer execution;
- NUMA and CPU affinity;
- huge-page behavior where supported;
- avoiding CPU reads of GPU-hot regions;
- overlapping model loading with initialization.

Measure page faults, residency, and effective bandwidth rather than assuming
zero-copy is faster.

## Quantization and Compression

### 11. Weight-only quantized decode

A device-native quantized GEMV may provide a larger decode improvement than an
isolated attention optimization because decode repeatedly streams model weights.

Suggested progression:

1. BF16 reference.
2. INT8 weight-only GEMV.
3. One groupwise low-bit format such as INT4/INT5/INT6.
4. Fused dequantization plus GEMV.
5. Per-tensor or per-layer mixed precision based on measured sensitivity.

GGUF storage layout does not have to be the execution layout. Repack weights at
load time into a gfx1151-specific representation when the one-time cost is
justified.

### 12. KV-cache compression

Explore FP16, BF16, INT8, or FP8-like cache representations with explicit
per-head or per-block scaling. Measure:

- attention error;
- final logit error;
- exact-token agreement;
- long-context retrieval quality;
- conversion overhead;
- effective memory bandwidth.

Cache precision must be part of backend support and cache-coherence contracts.

### 13. Mixed-precision recurrent state

Evaluate reduced-precision DeltaNet/SSM state separately from attention KV. The
state may have different numerical sensitivity and reuse characteristics.
Maintain an FP32 reference path and long-generation stability tests.

## External Kernel Technologies

### 14. Native HIP as the production baseline

Native HIP is currently the best fit for stable production integration:

- direct C++ integration;
- explicit gfx1151 specialization;
- complete control over wave and memory layout;
- no Python runtime dependency;
- straightforward fallback ownership.

### 15. Composable Kernel

Use CK where the exact operation and architecture are supported and measured.
Do not assume CDNA-tuned kernels are optimal on gfx1151. Preserve native
fallbacks and record rejected support predicates.

### 16. Triton as a prototyping tool

Triton can accelerate exploration of:

- tile sizes;
- split-K algorithms;
- memory layouts;
- fused epilogues;
- quantized kernels.

A practical workflow is:

```text
prototype in Triton
  -> benchmark on gfx1151
  -> identify a winning algorithm/layout
  -> port the stable path to native HIP
  -> integrate through the C++ dispatcher
```

A permanent Triton runtime dependency should be accepted only if it provides a
clear maintained advantage.

### 17. AITER as an oracle and source of experiments

AITER support for gfx1151 should be treated as experimental. Triton, FlyDSL, and
some HIP paths may be useful, while many CK/assembly kernels are oriented toward
CDNA. Use AITER for comparison and algorithm discovery before adopting it as a
runtime dependency.

## HRX System Runtime and Driver

### Current repository state

`.devops/nix/hrx-system.nix` provides a pinned, reproducible build of
`ROCm/hrx-system` at revision
`8c274e544a9d411eb05e1eb5629101dd44d2379e`.

The derivation:

- builds with ROCm LLVM/Clang;
- pins FlatCC, HSA runtime headers, and HIP API headers;
- enables `IREE_HAL_DRIVER_AMDGPU`;
- enables `IREE_HAL_DRIVER_HIP`;
- enables `LIBHRX_BUILD`;
- disables Loom;
- disables upstream tests and benchmarks;
- targets x86-64 Linux.

At present it is only exposed from `.devops/nix/scope.nix`. It is not a build
input of the Strix package, is not linked into the engine, and is not used by the
current HIP execution path. Therefore, the derivation alone cannot improve
performance.

### Performance hypothesis

HRX may provide a lower-level or IREE-oriented execution path with different
command submission, dispatch, synchronization, executable loading, and runtime
overheads than the conventional HIP path. Potential benefits should be tested in
areas where runtime overhead matters:

- many small decode kernels;
- command-buffer reuse;
- graph-like repeated execution;
- direct AMDGPU versus HIP HAL behavior;
- asynchronous scheduling;
- executable and pipeline caching;
- CPU submission overhead.

Do not describe HRX as faster until the same kernels and workloads are measured
under controlled conditions. It may improve dispatch overhead without improving
kernel execution, or dependency/runtime costs may outweigh its benefit.

### Expected benefit and adoption criteria

HRX is most likely to help decode workloads composed of many short kernels. It
could reduce CPU-to-runtime transitions, kernel-submission overhead,
synchronization costs, and GPU idle gaps between launches. Command-buffer reuse,
executable caching, direct AMDGPU dispatch, or IREE-driven scheduling and fusion
could make the benefit larger than a simple launch-latency improvement.

HRX is less likely to materially improve large prefill GEMMs or long-running
attention kernels, where GPU arithmetic and memory traffic dominate runtime. It
also cannot by itself fix weight-bandwidth-limited GEMV, inefficient attention
layouts, quantization overhead, low occupancy, or unnecessary cache traffic.
Those require better kernels, layouts, precision, or fusion.

Expected potential by workload:

| Workload | HRX potential | Main limitation |
| --- | --- | --- |
| Large prefill GEMMs | Low | GPU execution dominates submission overhead |
| Long-running prefill attention | Low | Kernel arithmetic and memory traffic dominate |
| Many small decode kernels | Medium to potentially high | Depends on measured GPU idle gaps |
| CPU launch-bound execution | Potentially high | Must beat HIP Graph capture |
| Weight-bandwidth-bound decode | Limited | Runtime changes do not reduce weight traffic |
| IREE-fused decode sequence | Potentially high | Requires significant compiler/runtime integration |
| GPU/NPU heterogeneous scheduling | Promising research path | Synchronization and memory contention may dominate |

HIP Graphs should be evaluated before introducing a complete HRX backend. Graph
capture preserves the existing HIP kernels, allocations, profiling, and device
compatibility while targeting much of the same repeated-launch overhead. If HIP
Graphs remove most inter-kernel gaps, HRX may not justify its additional API,
dependency, debugging, and maintenance surface. HRX becomes more compelling if
it still provides lower submission cost, better scheduling, useful fusion, or
better gfx1151/XDNA2 coordination.

The first controlled comparison should exercise four paths:

1. ordinary HIP;
2. HIP Graph capture;
3. the HRX HIP driver;
4. the HRX AMDGPU driver.

Measure empty submission, one small kernel, repeated small kernels, a complete
decode layer, and a complete generated token. Record CPU time, GPU idle time,
inter-token latency, tokens per second, initialization cost, and warm-cache
behavior. A favorable no-op or launch microbenchmark is insufficient; adoption
requires a repeatable end-to-end improvement on representative model workloads.

A major HRX development lane is justified only when at least one of these is
confirmed on gfx1151:

- decode spends a material fraction of time in submission-induced GPU gaps;
- HRX command reuse materially outperforms HIP Graphs;
- IREE produces a measurably better fused or scheduled decode sequence;
- HRX improves coordination between gfx1151 and XDNA2;
- the gain remains after GEMV, attention, and quantization improvements.

Until then, HRX should remain an experimental decode backend rather than a
global HIP replacement.

### Proposed HRX evaluation stages

#### Stage 1: package and capability validation

- Build `hrx-system` on the supported Nix machine.
- Record installed libraries, headers, tools, and CMake package metadata.
- Enable its own tests and benchmarks in a separate experimental derivation when
  feasible.
- Confirm gfx1151 device discovery and executable loading.
- Record required kernel, firmware, ROCm, and environment versions.

#### Stage 2: isolated microbenchmarks

Compare HRX AMDGPU, HRX HIP, and current HIP paths for:

- empty/no-op submission latency;
- repeated small kernel launch latency;
- event and synchronization latency;
- buffer allocation/import;
- host-mapped and device-resident memory;
- command-buffer reuse;
- one representative RMSNorm, GEMV, and elementwise kernel.

Use identical device code and buffers wherever possible.

#### Stage 3: optional backend boundary

Do not replace HIP globally. Introduce an experimental execution boundary only
after Stage 2 shows a real benefit:

```cpp
enum class GpuRuntime {
  kHip,
  kHrxHip,
  kHrxAmdgpu,
};
```

The production default remains HIP. Runtime selection must be explicit and
reported by diagnostics.

#### Stage 4: decode-sequence experiment

Port or wrap one stable decode sequence and compare:

- launches per token;
- CPU submission time;
- GPU idle gaps;
- inter-token latency;
- tokens/s;
- memory bandwidth;
- initialization and cache-warmup cost.

Only then decide whether HRX should become a supported runtime backend, a build
or kernel tool, or remain a research dependency.

### HRX risks

- rapidly changing API and pinned unstable revision;
- unclear gfx1151 maturity;
- IREE integration cost in a native C++ engine;
- duplicate HIP/HSA/runtime dependencies;
- driver and firmware compatibility;
- debugging and profiling tooling gaps;
- no tests or benchmarks enabled in the current derivation;
- risk of optimizing launch overhead while weight bandwidth remains dominant;
- additional maintenance surface for agent-driven development.

## Speculative Decoding

### 18. Conventional draft-target speculation

Implement ordinary speculative decoding before DFlash. This establishes the
hard runtime contracts:

1. A draft backend proposes `k` tokens.
2. The target verifies the block in one batched pass.
3. Tokens are accepted until rejection.
4. KV and recurrent state are committed or rolled back.
5. Draft length adapts to acceptance and verification cost.

Required capabilities:

- transactional KV state;
- transactional DeltaNet/SSM state;
- verifier logits for all proposed positions;
- exact acceptance/rejection semantics;
- tokenizer and vocabulary compatibility;
- rollback benchmarks;
- acceptance and speed telemetry.

### 19. Adaptive draft length

Select draft length using:

- recent acceptance rate;
- verifier batch efficiency;
- context length;
- server load;
- draft/target latency ratio;
- rollback cost;
- SSM state cost.

Optimize accepted target tokens per unit time, not raw draft tokens per second.

### 20. Self-speculative decoding

Possible approaches include layer skipping, early exit, sparse attention, or
using a cheaper model subgraph. Qwen-style sequential hybrid attention/SSM
architectures may have low acceptance when only one component is used as the
drafter, so this requires measurement rather than assumption.

### 21. Draft heads

Research Medusa-, EAGLE-, ReDrafter-, Gumiho-, or PARD-style multi-token heads.
These approaches require compatible trained weights or a training/export path,
but can avoid running a complete second model.

## DFlash and DFlash2

DFlash is block-diffusion speculative decoding, not a FlashAttention kernel. A
small block-parallel draft model predicts several future tokens, and the target
model verifies them. The formal published reference currently available is
DFlash; “DFlash2” should be treated as a model or implementation variant until a
stable independent specification is identified.

DFlash integration requires:

- a compatible trained draft checkpoint;
- target hidden-state extraction;
- target-to-draft feature transfer;
- block-causal or non-causal draft attention;
- parallel block proposal;
- batched target verification;
- cheap KV and recurrent-state rollback;
- model-specific import/export support.

Suggested interface:

```cpp
struct DraftBlock {
  std::vector<TokenId> tokens;
};

class DraftBackend {
 public:
  virtual ~DraftBackend() = default;
  virtual DraftBlock Propose(const VerifiedPrefix&, std::size_t max_tokens) = 0;
};
```

Possible implementations include autoregressive draft, multi-token head,
DFlash, and NPU draft backends. Production code may use static dispatch rather
than virtual calls; the interface above expresses the architectural contract.

## XDNA2 Opportunities

### 22. XDNA2 draft model plus gfx1151 verifier

A distinctive Strix Halo experiment is:

```text
XDNA2 NPU: small draft or block-diffusion model
      -> proposed tokens
 gfx1151 GPU: target batched verification
      -> accepted tokens
```

Potential advantages:

- draft and target use different compute resources;
- GPU remains dedicated to the large target model;
- unified memory may reduce explicit copies;
- draft and target work may overlap.

Risks:

- XRT invocation and synchronization overhead;
- NPU operator and model support;
- tensor-layout conversion;
- unified-memory bandwidth contention;
- hidden-state transfer cost for DFlash-like approaches;
- limited benefit when acceptance is low.

The key metric is draft-plus-verification time per accepted target token.

### 23. NPU preprocessing or auxiliary models

If speculative drafting is not initially viable, evaluate XDNA2 for:

- embeddings or lightweight encoders;
- reranking;
- safety/classification heads;
- prompt feature extraction;
- tokenizer-adjacent neural work;
- small model routing.

This can establish stable XRT scheduling and memory-sharing infrastructure before
placing generation-critical work on the NPU.

## Hybrid SSM/Attention Research

### 24. Replayable recurrent state

Speculation affects recurrent state as well as KV cache. Copying every layer’s
complete state for each speculative branch may eliminate the speedup.

Explore a ReplaySSM-like design:

- checkpoint state at verified boundaries;
- retain recent SSM inputs;
- replay accepted inputs after rejection;
- choose copy versus replay based on state size and draft length;
- fuse replay where possible.

### 25. Batched and persistent DeltaNet recurrence

Explore:

- register-cached state across token blocks;
- persistent recurrence kernels;
- chunk-size-specific kernels;
- state layout optimized for wave32;
- fused convolution, recurrence, normalization, and gate where occupancy allows;
- replay-friendly input logging for speculative decoding.

### 26. Component-aware hybrid speculation

Evaluate whether attention layers, SSM layers, early layers, or skipped-layer
paths can produce useful draft predictions. Sequential hybrid architectures may
behave differently from parallel hybrids, so acceptance must be measured by
layer type and draft length.

## Sparse and Long-Context Research

### 27. Sparse attention

Research static, dynamic, sliding-window, sink-token, or learned sparse patterns
for very long contexts. Requirements:

- explicit quality evaluation;
- retrieval and needle tests;
- sparse KV layout;
- verifier compatibility;
- stable fallback to exact attention.

### 28. Sparse speculative verification

Sparse drafting and full or sparse target verification may reduce KV bandwidth
at long context. The difficulty is maintaining compatible sparse layouts across
draft positions and efficiently verifying a block.

### 29. Prefix and retrieval-aware execution

Explore prefix reuse, prompt chunk caching, and retrieval-aware KV placement for
server workloads with repeated system prompts or shared documents.

## Suggested Execution Roadmap

### Phase A: correctness and measurement

- Fix or explicitly constrain FP16 cache transitions.
- Add kernel benchmarks and dispatch telemetry.
- Establish stable pp/tg/context baselines.
- Add dedicated gfx1151 CI or a controlled benchmark machine.

### Phase B: low-risk runtime and kernel work

- GEMV autotuning.
- hipBLASLt plan persistence.
- HIP Graph decode capture.
- targeted fusion.
- Flash-Decoding at long contexts.

### Phase C: compression

- one gfx1151-native weight-only format;
- fused dequantized GEMV;
- optional KV and recurrent-state compression.

### Phase D: HRX evaluation

- build and enable HRX tests/benchmarks;
- compare submission and synchronization overhead;
- test one identical kernel sequence;
- add an experimental backend only if measurements justify it.

### Phase E: speculative runtime

- conventional draft-target verification;
- transactional KV and SSM state;
- adaptive draft length;
- replayable recurrent state.

### Phase F: heterogeneous and experimental decoding

- XDNA2 draft plus GPU verifier;
- multi-token draft heads;
- DFlash/DFlash2;
- sparse self-speculation;
- component-aware hybrid speculation.

## Recommended First Experiments

1. Fix and test the baseline-to-tiled cache transition.
2. Benchmark GEMV bandwidth and launch overhead by projection shape.
3. Measure ordinary versus HIP Graph decode.
4. Prototype split-K Flash-Decoding at 4K, 8K, and 16K context.
5. Benchmark one INT8 weight-only GEMV against BF16.
6. Build HRX with tests/benchmarks enabled and measure no-op plus small-kernel
   submission latency against HIP.
7. Implement a minimal conventional speculative verifier before selecting a
   DFlash or NPU draft strategy.

## External References

- DFlash: <https://arxiv.org/abs/2602.06036>
- AngelSpec DFlash notes:
  <https://github.com/Tencent/angelspec/blob/main/docs/concepts/dflash.md>
- FlashAttention-2: <https://arxiv.org/abs/2307.08691>
- AMD AITER: <https://github.com/ROCm/aiter>
- ROCm model acceleration libraries:
  <https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-acceleration-libraries.html>
- HRX system runtime: <https://github.com/ROCm/hrx-system>
- ReplaySSM discussion: <https://github.com/vllm-project/vllm/issues/47572>
- MagicDec: <https://arxiv.org/abs/2408.11049>
- ReDrafter: <https://arxiv.org/abs/2403.09919>
- Gumiho: <https://arxiv.org/abs/2503.10135>
- PARD: <https://arxiv.org/abs/2504.18583>
