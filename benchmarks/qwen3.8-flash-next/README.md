# Qwen3.8-Flash-Next on Strix Halo

Linux x86-64, AMD `gfx1151`, 128 GB unified memory. Artifact:
`unsloth/Qwen3.8-Flash-Next-GGUF`, revision
`38bb39ee97821de2c9009abb7e93950eec396e66`, `UD-Q4_K_XL` (four shards).
MTP: `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` from the same revision.
Target weights occupy about 77 GiB of GPU memory; the 26.8 GiB n-gram table
stays on disk with about 6 MiB of compressed rows cached in RAM.
Context, scratch and MTP add memory. NVMe loading takes about
15 seconds; 256 MiB of upload staging is released before inference.

## Memory

C1, context limit 262144, one server session, a 3891-token repetitive prompt,
128 generated tokens and seed 1. GPU-visible unified allocations use the Linux
GTT counter with the idle host subtracted; separate CPU memory is not included.

| Mode | Loaded GiB | After generation GiB |
| --- | ---: | ---: |
| AR | 85.39 | 85.66 |
| MTP | 89.72 | 90.01 |

The full requested KV capacity remains allocated. Raw indexer keys use a
4096-row ring; completed blocks retain their pooled keys. PLE, MTP and trunk
activations share storage when their lifetimes do not overlap. Snapshots keep
only unpooled raw keys; existing disk cache entries rebuild after this layout
change. MTP adds a 0.33 GiB Q4 shortlist head; target weights and cache
precision are unchanged.

## Performance

Nix release, C1, pp2048, tg128, seed 1, one measured repetition,
2026-09-18 UTC.
The CLI uses deterministic repetitive text, warms prefill
and continues generation past EOS. Depth precedes the measured operation.
Rates are tok/s; profiler timings are excluded.

| Depth | AR pp2048 | MTP pp2048 |
| ---: | ---: | ---: |
| 0 | 1571.61 | 1555.39 |
| 4096 | TODO | TODO |
| 16384 | TODO | TODO |
| 32768 | 1423.63 | 1305.29 |
| 65536 | TODO | TODO |
| 131072 | TODO | TODO |

| Sampling | Depth | AR tg128 | MTP tg128 | MTP acceptance |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 0 | 26.24 | 47.89 | 71.0% |
| Greedy | 32768 | 24.04 | 59.17 | 100% |
| Temperature 0.7 | 0 | TODO | 47.86 | 75.9% |
| Temperature 1.0, top-p 0.95 | 0 | TODO | TODO | TODO |

Greedy AR and MTP use the same 34,817-token context limit at both depths;
the separate temperature 0.7 run uses 4,096. MTP PP includes draft catch-up.
TG starts from 16 prompt tokens at d0 and from the specified depth otherwise.
The d32K continuation accepted all 110 draft proposals; acceptance depends
on the prompt and depth. SSM convolution and QKV normalization/cache writes
are fused into projections for prefill batches of at least 1024 tokens.
The HC down projection fuses its activation and F16 conversion from 96 tokens.
The targets are **1700 tok/s pp2048** and near-flat PP through d128K;
both remain TODO.
HTTP, C1, context 4096, seed 1, up to 128 output tokens, uncached prompts:

| Prompt | MTP greedy tok/s | MTP temperature 0.7 tok/s |
| --- | ---: | ---: |
| Repetitive pattern | 82.01 | 73.49 |
| Python iterator merge | 48.30 | 48.25 |
| Probability exercise | 58.74 | 52.52 |

The sampled repetitive response ended at 78 tokens; other rows generated 128.
Concurrent throughput: TODO.
Serving interleaves sessions but does not batch model work;
`gufo bench` supports C1 for this model.

```sh
./result/bin/gufo bench --model "$MODEL" -p 2048 -n 128 \
  -d 0,4096 --temperature 0 --seed 1 -v
./result/bin/gufo bench --model "$MODEL" -p 2048 -n 128 \
  -d 0,4096 --temperature 0.7 --seed 1 -v \
  --speculative mtp --mtp-model "$MTP"
```

Internal prefill chunks use 2048 tokens, selected by a 512–4096 sweep.
HTTP accepts arbitrary prompt lengths and yields between chunks. PLE reads
overlap layer 0; its bounded cache retains original compressed rows.
Small cached gathers complete without waking the I/O workers; large gathers
still decode rows in parallel. Expert-count downloads precede the shared
expert, allowing CPU tile selection to overlap its GPU projections.
The [projection sweep](tools/projection_plans.hip) uses `tools/bench/build.sh`;
`tools/prof/prof.py run --stages qwen-flash --` profiles a release command
inside `nix develop`.

`prompt`, `chat` and `serve llm` share MTP and sampling options. A private Q4
head shortlists 256 vocabulary rows, then the original Q8 head rescores them.
Greedy takes the best rescored row; temperature sampling draws from the best
64 using the request's filters and penalties. GPU verification uses
the full target vocabulary: accept with `min(1, p/q)`, otherwise draw from
normalized `max(p-q, 0)`. The correction becomes the next batch's anchor.
Compact draft support never truncates the target. Shortlisting can alter draft
support; it is not guaranteed to reproduce the full Q8 head's top 64 on every
input. Rescored logits use the full Q8 kernel's exact arithmetic.

Adaptive MTP is the default; `--draft-tokens` caps the chain at 1–7 drafts.
Decisions use committed acceptance history and reset with the session.
Snapshots preserve this state; new HTTP requests reset it when reusing
context. Decisions never depend on wall-clock timings.

Retained optimizations: fused prefill projections, padded WMMA transposes,
grouped expert verification, paired F16 expert stores, Q4 shortlisting with
Q8 rescoring, packed MTP readbacks and GPU verification with one frontier
readback. Plain greedy verification returns GPU argmax results; penalties
retain the shared sampler's semantics. Sampled verification preserves the
original F32 sum order.

The indexer prefetches keys for batches of at least 256 queries. Exact top-k
uses a finer histogram window at 16K depth and above, falling back to the full
range if the threshold is clipped. Every score participates.
Larger attention tiles, wider DeltaNet reductions and carrying convolution
boundaries across larger tiles were slower. Compact attention scratch gave
no material gain. Sparse attention remains the largest measured depth penalty.
Blocking HIP waits reduced CPU use but slightly slowed generation, so were
not retained. Sampling and exponentials accounted for about 2% of CPU cycles
in unfiltered temperature 0.7 AR; most CPU cycles were spent waiting for the GPU.

## Quality checks

Tests and probes live in `tests/models/qwen38_flash_next/`. The
`qwen38_flash_next_tests` build target contains 13 checks: n-gram I/O, MTP
sampling, and 11 GPU operator/upload checks. Each covers a separate contract.
Full-model tests are in the explicit `qwen38_flash_next_model_tests` build
target; run `session_test` for decoding/sampling/wiring and `snapshot_test`
for persistence and rollback. They require model paths and are not automatic
CTest jobs. CPU/reference and GPU logit probes are explicit diagnostic targets;
they do not implement a separate speculative generation loop.

Build and run the operator affected by a change first; for example:

```sh
nix develop -c cmake --build --preset gpu-test --target qwen38_flash_next_moe_ids_ops_test
nix develop -c ctest --test-dir build/gpu-test -R 'qwen38_flash_next[.]moe_ids_ops' \
  --no-tests=error --output-on-failure
```

- **Operators:** independent numerical references, FP64 projection dots,
  exact outputs and replay. Cover ragged shapes, guards, inactive experts,
  finite outputs, tied selections, indexer ring wraps, and attention near 128K.
  Router tests check
  softmax weights and lowest-index ties independently of assignment maps.
  Fusions must preserve the separate operators' rounding. Decode and
  verification projections must agree across batch widths one through eight;
  routed down projections also check expanded top-10 batches. Grouped
  projections cover duplicate expert slots within and across tokens. MTP head
  checks cover Q4 quantization, bit-exact selected Q8 rows, lowest-ID ties
  including signed zero, nonfinite logits, packed workspace reuse and graph
  replay. Indexer checks compare every mask bit with a CPU full sort, including
  both histogram fallback tails and a 257-query batch near 128K depth.
- **Model replay:** greedy AR/MTP tokens, full logits, RNG and positions must
  match at short and 4K contexts. Sampled MTP must replay itself exactly;
  teacher-forcing its tokens through AR must reproduce every frontier logit.
  Cover fresh loads, request order, resets, snapshots and rollback prefixes.
  Build `qwen38_flash_next_model_tests`, then run
  `build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_session_test
  --model "$MODEL" --mtp-model "$MTP"` for retained model changes. Run the
  sibling `qwen38_flash_next_snapshot_test` with the same arguments when
  state or snapshot handling changes.
- **Sampling and wiring:** the shared 23-case serving matrix covers AR/MTP,
  filters, penalties, replay, C2 and short output budgets. Direct session and
  serving outputs must agree. Proposal tests check exact exported probability
  masses, target output frequencies and deferred-draw replay. The shared GPU
  sampler checks acceptance/residual CDF boundaries, duplicate candidates and
  target mass outside draft support. Wiring changes also require actual
  prompt, chat and sampled HTTP requests.
- **Loading and I/O:** verify upload bytes, guards, shard/chunk boundaries,
  exact IQ4_NL/BF16 cold, cached and mixed n-gram reads, duplicate/reordered
  rows, asynchronous completion, cache collisions and eviction, failed I/O
  recovery, cached reads after truncation, and teardown.

Prefill changes also require `gufo bench --validate-prefill N` and CPU/reference
probes. Benchmark retained changes with the Nix release and the same artifact,
context capacity and request setup. Keep exact output and acceptance checks
beside speed results; a profiler trace is not a speed benchmark.
Standalone kernel ablations must match production's HIP flags:
`-O3 -ffast-math -fno-finite-math-only`; masked softmax relies on infinities.
The current 1024-token prefill check is finite, with scalar-winner rank 1,
logit RMSE 0.24 and maximum absolute error 1.15. Snapshot compatibility
includes inference arithmetic; older cached states rebuild after it changes.

Sampled MTP preserves the target sampling distribution within floating-point
precision, but need not produce AR's same-seed tokens or a common prefix
across different output budgets. Greedy decoding retains those guarantees.
Batched prefill differs numerically from sequential decoding. AR/MTP agreement
does not establish unquantized upstream equivalence. Full-model upstream
quality qualification remains TODO.
