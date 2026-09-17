# Qwen3.8-Flash-Next on Strix Halo

Linux x86-64, AMD `gfx1151`, 128 GB unified memory. Artifact:
`unsloth/Qwen3.8-Flash-Next-GGUF`, revision
`38bb39ee97821de2c9009abb7e93950eec396e66`, `UD-Q4_K_XL` (four shards).
MTP: `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` from the same revision.
Target weights occupy about 77 GiB of GPU memory; the 26.8 GiB n-gram table
stays on disk with about 6 MiB of compressed rows cached in RAM.
Context, scratch and MTP add memory. NVMe loading takes about
15 seconds; 256 MiB of upload staging is released before inference.

## Performance

Nix release `8e60594`, C1, pp2048, tg128, seed 1, one measured repetition
(2026-09-17 UTC). The CLI benchmark uses deterministic repetitive text.
Prefill is warmed before measurement; depth precedes the measured operation.
Rates are tok/s. Fixed-length generation continues past EOS.

| Depth | AR pp2048 | MTP pp2048 |
| ---: | ---: | ---: |
| 0 | 1462.41 | 1501.23 |
| 4096 | TODO | 1372.32 |
| 16384 | TODO | TODO |
| 32768 | 1378.66 | TODO |
| 65536 | 1340.91 | TODO |
| 131072 | 1304.59 | TODO |

| Sampling | Depth | AR tg128 | MTP tg128 | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 0 | 26.19 | 42.83 | 71.0% |
| Greedy | 4096 | 25.22 | 36.66 | 63.7% |
| Temperature 0.7 | 0 | 25.56 | 35.76 | 65.5% |
| Temperature 0.7 | 4096 | 24.86 | 49.84 | 93.0% |
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

`prompt`, `chat` and `serve llm` share MTP and sampling options. Drafts are
greedy over the full vocabulary on the GPU; verification uses the target
sampler with its filters, penalties and RNG. Adaptive MTP is the default;
`--draft-tokens` caps the chain at 1–7 drafts. The controller uses committed
acceptance history, counts only the first rejection as a failure, and resets
with the session. Decisions never depend on wall-clock timings.

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
  finite outputs, tied selections, and attention near 128K. Router tests check
  softmax weights and lowest-index ties independently of assignment maps.
  Fusions must preserve the separate operators' rounding. Decode and
  verification projections must agree across batch widths one through eight;
  routed down projections also check expanded top-10 batches.
- **Model replay:** compare AR/MTP tokens, full logits, RNG and positions at
  short and 4K contexts, including fresh loads, request order, draft widths
  and resets. Rollback tests cover every proper prefix and full-batch commit.
  Run `build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_session_test
  --model "$MODEL" --mtp-model "$MTP"` for retained model changes.
- **Sampling and wiring:** the shared 23-case serving matrix covers AR/MTP,
  filters, penalties, replay, C2 and short output budgets. Verifier tests cover
  accepted/rejected prefixes, stops and RNG rollback. Wiring changes also
  require actual prompt, chat and sampled HTTP requests.
- **Loading and I/O:** verify upload bytes, guards, shard/chunk boundaries,
  asynchronous n-gram reads, duplicate/reordered rows, cache collisions and
  eviction, failed I/O recovery, and teardown.

Prefill changes also require `gufo bench --validate-prefill N` and CPU/reference
probes. Benchmark retained changes with the Nix release and the same artifact,
context capacity and request setup. Keep exact output and acceptance checks
beside speed results; a profiler trace is not a speed benchmark.

Batched prefill differs numerically from sequential decoding. AR/MTP agreement
does not establish unquantized upstream equivalence. Full-model upstream
quality qualification remains TODO.
