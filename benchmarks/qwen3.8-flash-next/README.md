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

| Mode | Loaded GiB | After generation GiB | Greedy TG tok/s | Temperature 0.7 TG tok/s |
| --- | ---: | ---: | ---: | ---: |
| AR | 85.39 | 85.66 | 25.70 | 25.19 |
| MTP | 89.39 | 89.67 | 59.82 | 56.99 |

The full requested KV capacity remains allocated. Raw indexer keys use a
4096-row ring; completed blocks retain their pooled keys. PLE, MTP and trunk
activations share storage when their lifetimes do not overlap. Snapshots keep
only unpooled raw keys; existing disk cache entries rebuild after this layout
change. Weights, cache precision and sampling are unchanged.

## Performance

Nix release builds, C1, pp2048, tg128, seed 1, one measured repetition
(2026-09-17 UTC). The CLI benchmark uses deterministic repetitive text.
Prefill is warmed before measurement; depth precedes the measured operation.
Rates are tok/s. Fixed-length generation continues past EOS. AR measurements
are from `8e60594`; MTP measurements include sampled proposals.

| Depth | AR pp2048 | MTP pp2048 |
| ---: | ---: | ---: |
| 0 | 1462.41 | 1496.14 |
| 4096 | TODO | 1372.44 |
| 16384 | TODO | TODO |
| 32768 | 1378.66 | TODO |
| 65536 | 1340.91 | TODO |
| 131072 | 1304.59 | TODO |

| Sampling | Depth | AR tg128 | MTP tg128 | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 0 | 26.19 | 43.04 | 71.0% |
| Greedy | 4096 | 25.22 | 36.61 | 63.7% |
| Temperature 0.7 | 0 | 25.56 | 43.48 | 75.9% |
| Temperature 0.7 | 4096 | 24.86 | 38.89 | 67.7% |
| Temperature 1.0, top-p 0.95 | 0 | TODO | TODO | TODO |
| Temperature 1.0, top-p 0.95 | 4096 | TODO | TODO | TODO |

Other depth and concurrent throughput measurements: TODO. Serving interleaves
sessions but does not batch model work; `gufo bench` supports C1 for this model.
AR PP uses one 133,121-token context limit throughout; AR TG uses 4,225
and all MTP measurements use 6,145. The target is **1700 tok/s pp2048**;
reaching it and retaining near-flat throughput through d128K remain TODO.
The measured d0-to-d128K decline is 10.8%.
At d32K, the latest profile attributes about 763 ms to expert/dense
projections and 147 ms to attention, out of 1481 ms GPU time.

HTTP prefill controls use 2048-token excerpts of `docs/PERFORMANCE.md` and
`src/models/qwen38_flash_next/ngram.cpp` at `3ee6aea`, context 4096,
one untimed warm request, and one output token. All have zero cached prefix
tokens. Revised requests change the beginning but reuse most PLE rows.

| HTTP input | PP tok/s |
| --- | ---: |
| Documentation | 1362.48 |
| Code | 1359.27 |
| Revised documentation | 1403.44 |
| Revised code | 1395.72 |

HTTP MTP generation, context 6145, temperature 0.7, seeds 1–4, one warm
request per input and a reused prompt cache. Each measured request emits
128 tokens. Rates divide total output tokens by total decode time.

| HTTP input | Prompt tokens | TG tok/s | Acceptance |
| --- | ---: | ---: | ---: |
| Repetitive completion | 3891 | 57.32 | 100.0% |
| Python merge function, chat | 33 | 47.77 | 78.2% |

Sampling changes the generated path, so speed and acceptance vary by seed.
The CLI temperature-0.7 ranges over seeds 1–3 are 38.28–43.48 tok/s at d0
and 36.86–38.89 at d4K. Very short, EOS-limited replies are not steady
generation measurements.

```sh
./result/bin/gufo bench --model "$MODEL" -p 2048 -n 128 \
  -d 0,4096 --temperature 0 --seed 1 -v
./result/bin/gufo bench --model "$MODEL" -p 2048 -n 128 \
  -d 0,4096 --temperature 0 --seed 1 -v \
  --speculative mtp --mtp-model "$MTP"
```

Internal chunks use 2048 tokens, selected by a 512–4096 sweep. HTTP accepts
arbitrary prompt lengths and yields between chunks. Resident readers gather
PLE rows while layer 0 runs, including graph replay. Large gathers read in
file order; a bounded cache retains original compressed bytes.
Projection plans and exact sparse selections are deterministic.
The [projection sweep](tools/projection_plans.hip) uses `tools/bench/build.sh`;
`tools/prof/prof.py run --stages qwen-flash --` profiles a release command
inside `nix develop`. Profiler throughput is not a benchmark result.

`prompt`, `chat` and `serve llm` share MTP and sampling options. Greedy
decoding uses full-vocabulary argmax drafts. Temperature sampling draws
from 64 MTP candidates with the request's filters and penalties. GPU
verification uses the full target vocabulary: accept with `min(1, p/q)`,
otherwise sample normalized `max(p-q, 0)`. The correction becomes the next
batch's anchor. Compact draft support never truncates the target.

Adaptive MTP is the default;
`--draft-tokens` caps the chain at 1–7 drafts. The controller uses committed
acceptance history, counts only the first rejection as a failure, and resets
with the session. Snapshots preserve its state; new HTTP requests reset it
when reusing context. Decisions never depend on wall-clock timings.

## Quality checks

Tests and probes live in `tests/models/qwen38_flash_next/`. Build and run the
operator affected by a change first; for example:

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
  routed down projections also check expanded top-10 batches.
- **Model replay:** greedy AR/MTP tokens, full logits, RNG and positions must
  match at short and 4K contexts. Sampled MTP must replay itself exactly;
  teacher-forcing its tokens through AR must reproduce every frontier logit.
  Cover fresh loads, request order, resets, snapshots and rollback prefixes.
  Run `build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_session_test
  --model "$MODEL" --mtp-model "$MTP"` for retained model changes.
- **Sampling and wiring:** the shared 23-case serving matrix covers AR/MTP,
  filters, penalties, replay, C2 and short output budgets. Direct session and
  serving outputs must agree. Proposal tests check exact exported probability
  masses, target output frequencies and deferred-draw replay. The shared GPU
  sampler checks acceptance/residual CDF boundaries, duplicate candidates and
  target mass outside draft support. Wiring changes also require actual
  prompt, chat and sampled HTTP requests.
- **Loading and I/O:** verify upload bytes, guards, shard/chunk boundaries,
  asynchronous n-gram reads, duplicate/reordered rows, cache collisions and
  eviction, failed I/O recovery, and teardown.

Prefill changes also require `gufo bench --validate-prefill N` and CPU/reference
probes. Benchmark retained changes with the Nix release and the same artifact,
context capacity and request setup. Keep exact output and acceptance checks
beside speed results; a profiler trace is not a speed benchmark.

Sampled MTP preserves the target sampling distribution within floating-point
precision, but need not produce AR's same-seed tokens or a common prefix
across different output budgets. Greedy decoding retains those guarantees.
Batched prefill differs numerically from sequential decoding. AR/MTP agreement
does not establish unquantized upstream equivalence. Full-model upstream
quality qualification remains TODO.
