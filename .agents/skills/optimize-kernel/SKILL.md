---
name: optimize-kernel
description: Optimize Gufo inference on Strix Halo gfx1151 using focused profiles, independent quality checks, and matched end-to-end measurements.
metadata:
  origin: gufo
---

# Optimize gfx1151 kernels

Read the affected model's `docs/models/<model>/{BENCHMARKS,EVALUATION,EXPERIMENTS}.md`
and its launch code. Establish the timed scope and arithmetic contract before
changing dispatch. Follow the user's machine, time, and Git instructions.

## Fast experiment loop

1. Keep baseline and candidate production binaries. Use `nix build`, or the
   `release` CMake preset with the same compiler/dependency versions. Test
   binaries keep assertions and are for correctness, not headline timings.
2. Start with one affected shape and one control. Use pp2048/tg128 for retained
   end-to-end results; tiny prompts/outputs suffice to debug launches. For a
   depth issue, try d16K/d32K first. Expand to d64K/d128K only when justified.
3. Profile separately with `tools/prof/prof.py`; compare stage totals, launch
   counts, GPU-busy time and wall span. Inspect allocator, synchronization,
   sampling and cache I/O when GPU time does not explain latency.
4. Change one mechanism, compile the affected target, then run its existing
   analytic/operator check. Use `tools/bench/build.sh <name>` for standalone
   HIP experiments; each result needs correctness alongside throughput.
5. Compare unprofiled baseline/candidate under identical model, prompt, depth,
   quantization, sampling, cache history and concurrency. Repeat only to resolve
   noise. A microbenchmark gain must survive the full inference path.
6. Keep the winning implementation as default. Delete rejected/dead routes;
   do not add environment switches or duplicate tests to preserve experiments.
   Record one concise retained/rejected row in `EXPERIMENTS.md`, current speeds
   in `BENCHMARKS.md`, and actual quality evidence in `EVALUATION.md`.

## Techniques that worked here

- **Bandwidth and cache:** measure `tools/bench/gfx1151_peak.hip`, not a spec
  sheet. Previous large-buffer controls reached roughly 240 GB/s. The 32 MiB
  MALL cache can make repeatedly reused small matrices look unrealistically
  fast: rotate weights beyond cache capacity or reproduce full-model traffic.
- **GEMV:** coalesce packed weight loads, stage shared activations in LDS,
  fuse QKV or gate/up reads where useful, and preserve each dot product's
  accumulation order. More fusion can increase VGPR pressure and spills.
  Two-iteration prefetch helped the small 320×10240 HC projection; wider
  prefetch and applying it to larger matrices did not. Preserve the actual
  rounded products/FMA sequence, not just the algebraic formula.
- **Verification and concurrency:** reuse quantized weights across token and
  request rows. Tune real ragged shapes, including 3–8 rows and partial tiles.
  Batch the complete draft transformer body, not only the vocabulary head;
  keep each request's KV, recurrence, acceptance and RNG state independent.
  Test both shared and disjoint expert routing: weight reuse benefits shared
  experts, but single-request experts need a compact path.
  Exact packed-integer transforms can help both: spreading Q5 high-bit
  nibbles with multiply/mask removed shifts without changing dequantization.
- **Wave mode and compiler:** selected quantized kernels need wave64 while
  other paths use wave32. Match helper/caller wave modes per translation unit.
  Iterative ILP scheduling helped selected 16-row Q4/Q5 kernels; applying it
  globally was not a win. Inspect VGPR/LDS/private scratch and ISA with
  `tools/prof/isa_mix.py`; occupancy alone is not the optimization objective.
  For wide register-cached softmax, a scheduling barrier after each exponential
  removed spills while preserving the sum order. Cache only bounded row sizes.
- **Attention and selection:** exact partial top-k can avoid sorting the full
  context. Preserve tie ordering and FP32 ranking. Pack existing KV bytes into
  bounded scratch/head groups without changing persistent precision. Do not
  change softmax reduction order without the model's numerical qualification.
- **Fusion and memory:** split repeated embedding/hidden projections instead
  of concatenating duplicate inputs. Respect full-width HC normalization.
  Reuse scratch, allocate rollback depth on demand, and avoid carrying entire
  prefill chunks into draft catch-up. UMA still has placement costs: mapped
  quantized weights and copied reusable audio/DiT weights behave differently.
- **Graphs:** verify replay actually launches every new operation. Side-stream
  work during capture is not automatically included in the graph. Check the
  relevant shallow/deep dispatch boundaries and unprofiled wall time.

## Quality and reporting

Use an independent operator formula or pinned official teacher; two Gufo
paths agreeing does not prove upstream parity. Compare matched token histories
and find the first changed layer if logits drift. Never relax tolerances or
replace goldens to accept a speedup. Test finite outputs and awkward tails
(e.g. 1/8/9/32/33 rows), not only aligned shapes.

For state or speculative changes, cover greedy output, sampled p/q rejection
and residual correction, seeded replay, EOS, multi-turn continuation and
snapshot restore. Add images/cancellation when those paths change. Timing-based
controllers are for greedy decoding; sampled execution must retain seeded
replay. Full model sweeps are final qualification, not each iteration.

Measure C1 plus affected C2/C4/C6/C8. Keep prompt and cache state identical when
comparing CLI single-user and HTTP C1; report decode rate separately from whole
request throughput. Raise the benchmark server's per-client queue limit for
C8 from one host; a rejected request is not a throughput result. For H3,
prefer an analytic case or one block/forward pass;
do not generate full videos during routine kernel work.
