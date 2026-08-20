# Strix Halo Performance Engineering

Status: stable guidance, 2026-08-19.

This document defines the measurement and promotion rules for performance
work. Current model numbers belong in `benchmarks/<model>/README.md`; issue
comments and local artifacts hold detailed experiment history.

## Target

The only production target is Linux x86-64 on AMD Strix Halo:

- `gfx1151` GPU using Wave32 HIP kernels.
- XDNA2/AIE2P NPU.
- Unified LPDDR5X shared by CPU, GPU, and NPU.
- Nix-provided compilers, libraries, profilers, and test tools.

Runtime probes, not assumed peak specifications, determine the available CUs,
AIE columns, memory, firmware, power mode, and thermal state.

## Optimization Loop

1. Define the numerical and mutable-state contract.
2. Measure the complete workload and identify the dominant phase.
3. Classify it as bandwidth, compute, launch, synchronization, or scheduling
   limited.
4. Build a focused reproducer with a trusted oracle.
5. Change one mechanism at a time.
6. Reject candidates that lose focused correctness or measured performance.
7. Run broader quality and integration gates only for the retained candidate.
8. Record the current result, not the full experiment diary.

Optimize end-to-end request behavior. A faster isolated kernel is not a win if
packing, synchronization, state management, or thermals erase the gain.

## Measurement Contract

Comparable runs use the same:

- Source revision and Nix build mode.
- Model revision, artifact, quantization, and KV representation.
- Prompt tokens, generated-token count, batch, depth, and concurrency.
- GPU/NPU route and dispatch configuration.
- Driver, firmware, ROCm, XRT, power mode, and memory configuration.
- Warmup policy and repetition count.

Alternate baseline and candidate runs on the same machine. Report medians and
tail values rather than the best sample. Record the starting thermal state for
long model runs; do not attribute an increasing-length sweep when the shared
APU heats materially between cases.

Headline latency comes from an unprofiled run. Use separate profiler runs for
kernel timing, counters, occupancy, VGPR, LDS, and scratch because tracing
changes timing.

Benchmark artifacts may include a privacy-safe machine fingerprint. They must
not include hostnames, usernames, local paths, prompts, generated text,
credentials, process secrets, or raw token IDs.

## Workload Guide

| Workload | Typical limit | First measurements |
| --- | --- | --- |
| Single-token decode | Weight and KV bandwidth, launch count | Effective bytes/s, kernel count, context scaling |
| Prompt prefill | GEMM utilization, attention reuse, packing | Projection and attention share, batch scaling |
| Long-context attention | KV traffic and parallelism | Per-layer attention time at 4K/8K/12K/16K |
| Speculative verification | Checkpoint, rollback, acceptance | Accepted tokens, replay time, baseline token parity |
| HTTP serving | Admission, queueing, session reuse | TTFT, inter-token latency, cancellation cleanup |
| NPU offload | DMA, synchronization, padded work | Transfer time, program time, end-to-end overlap |

For decode GEMV, reduce bytes read before chasing peak matrix throughput. For
prefill GEMM, measure reuse and matrix-instruction utilization. For every
heterogeneous route, include shared-memory-bandwidth contention and transfer
cost in the result.

## Implementation Rules

### Host runtime

- Use RAII for mappings, streams, events, graphs, contexts, and allocations.
- Allocate model, KV, recurrent, graph, and scratch resources before serving.
- Avoid general heap allocation in decode dispatch.
- Keep hot scheduler data compact and ownership explicit.
- Prefer single-owner state machines and bounded message passing.
- Do not hold a process-wide mutex while waiting for a device.
- Reject size and offset overflow before allocating or launching.

### HIP

- Compile production kernels for `gfx1151` and treat Wave32 as explicit.
- Prove alignment before vector loads and provide bounded tail paths.
- Track VGPR, LDS, occupancy, and scratch for retained kernels.
- Keep distinct decode and prefill routes where their reuse differs.
- Read only live KV spans and reuse GQA/MQA K/V across query heads.
- Retain hipBLASLt or rocBLAS as the baseline for supported matrix shapes.
- Do not inherit launch geometry from CUDA or another RDNA target without a
  new measurement on `gfx1151`.

### XDNA2

- Include host packing, BO synchronization, program time, and output transfer.
- Reuse AIE programs, contexts, BOs, and command buffers.
- Keep memory-tile layouts explicit and reject unbounded padding.
- Prefer work that is independent of the critical GPU path.
- Promote concurrent GPU/NPU execution only after measuring shared-memory
  contention and request latency.

### Quantization

- Report artifact size and effective bits per weight, including metadata.
- Fuse unpacking, scale application, and zero correction when practical.
- Avoid materializing dequantized weights in global memory.
- Verify full-vocabulary logits before reporting speed.
- Use separate kernel strategies when metadata or group shape changes.

## Commands

### Build and quality gate

Nix must see new files, so stage them before building:

```sh
git add <changed-files>
nix build
nix build .#checks.x86_64-linux.pr
```

Run focused tests during iteration. Run the full hardware or full-logit suite
when a retained kernel, numerical route, state transition, or model dispatch
changes.

### End-to-end model benchmark

```sh
MODEL=models/<model>/<artifact>.gguf

./result/bin/strix-server bench \
  --model "$MODEL" \
  --n-prompt 128,512,1024,2048,4096 \
  --n-gen 0 \
  --repetitions 3

./result/bin/strix-server bench \
  --model "$MODEL" \
  --n-prompt 2048 \
  --n-gen 128 \
  --n-depth 4096,8192,12288,16384 \
  --repetitions 1

./result/bin/strix-server bench \
  --model "$MODEL" \
  --validate-prefill 1024 \
  --n-prompt 1024 \
  --n-gen 0 \
  --repetitions 1
```

Keep current per-model results and matching third-party commands in that
model's benchmark README.

### Focused HIP benchmark

List cases, then run the smallest relevant matrix:

```sh
./result/bin/strix-kernel-bench --list

./result/bin/strix-kernel-bench \
  --case decode-attention \
  --context 4096,8192,12288,16384 \
  --warmup 5 \
  --repetitions 30 \
  --json
```

The report includes raw HIP-event samples, summary latency, correctness
sentinels, dispatch choices, and the privacy-safe machine fingerprint.

### Profiling

```sh
nix develop -c rocprofv3 \
  --kernel-trace \
  --marker-trace \
  --scratch-memory-trace \
  --stats \
  --summary \
  --output-directory /tmp/strix-profile \
  -- ./result/bin/strix-kernel-bench \
    --case decode-attention \
    --context 16384 \
    --warmup 1 \
    --repetitions 3
```

Use the emitted ROCTx case marker to isolate the measured region. Add selected
PMC counters only in a separate diagnostic pass.

### Offline tuning and replay

```sh
./result/bin/tune_hipblaslt \
  --out /tmp/strix-hipblaslt-plans.bin \
  --warmup 3 \
  --repetitions 10

STRIX_HIPBLASLT_PLAN_CACHE=/tmp/strix-hipblaslt-plans.bin \
  ./result/bin/strix-server bench ...

./result/bin/benchmark_ssm_replay \
  --model "$MODEL" \
  --context 128 \
  --draft-lengths 1,2,4,8,16
```

Generated profiler, plan, and replay artifacts stay outside the repository.

### Dispatch telemetry

```sh
STRIX_DISPATCH_TELEMETRY=1 ./result/bin/strix-kernel-bench ...
```

Telemetry is diagnostic JSONL. It records semantic dispatch decisions and
must not contain prompts, tokens, model paths, or machine identity.

## Promotion Gates

A performance change is retained only when:

- Focused correctness passes against the canonical oracle.
- Full-vocabulary logits remain finite and within the accepted envelope.
- Greedy output keeps the required token parity.
- Request state, rollback, cancellation, and cleanup remain correct.
- The declared workload improves outside measurement noise.
- Memory, startup, and tail latency do not regress unexpectedly.
- Required Nix checks pass on the exact staged source.

Document the winning route, current measurements, reproduction command, and
remaining gap. Keep rejected alternatives to a short summary or the relevant
issue discussion.
