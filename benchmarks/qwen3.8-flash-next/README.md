# Qwen3.8-Flash-Next on Strix Halo

Linux x86-64, AMD `gfx1151`, 128 GB unified memory. Artifact:
`unsloth/Qwen3.8-Flash-Next-GGUF`, revision
`38bb39ee97821de2c9009abb7e93950eec396e66`, `UD-Q4_K_XL` (four shards).
MTP: `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` from the same revision.
Target weights occupy about 77 GiB of GPU memory; the 26.8 GiB n-gram table
stays on disk. Context, scratch and MTP add memory. NVMe loading takes about
15 seconds; 256 MiB of upload staging is released before inference.

## Performance

Nix release, C1, pp2048, tg128, seed 1, one measured repetition
(2026-09-17 UTC). An aligned full-chunk context prefix also warms prefill;
other shapes use a separate untimed pass. Rates are tok/s. Depth precedes the measured operation;
fixed-length generation continues past EOS. MTP scores the full vocabulary
and adapts between one and seven drafts from committed acceptance history.

| Depth | AR pp2048 | MTP pp2048 |
| ---: | ---: | ---: |
| 0 | 1373.01 | 1387.23 |
| 4096 | TODO | 1277.06 |
| 16384 | TODO | TODO |
| 32768 | TODO | TODO |
| 131072 | 1225.96 | TODO |

| Sampling | Depth | AR tg128 | MTP tg128 | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 0 | 26.13 | 42.37 | 71.0% |
| Greedy | 4096 | 25.25 | 36.28 | 63.7% |
| Temperature 0.7 | 0 | TODO | 36.88 | 65.5% |
| Temperature 0.7 | 4096 | TODO | 49.57 | 93.0% |
| Temperature 1.0, top-p 0.95 | 0 | TODO | TODO | TODO |
| Temperature 1.0, top-p 0.95 | 4096 | TODO | TODO | TODO |

Other depth and concurrent throughput measurements: TODO. Serving interleaves
sessions but does not batch model work; `gufo bench` supports C1 for this model.
AR PP uses one 133,121-token context limit throughout; AR TG uses 4,225
and MTP PP/greedy TG use 6,145 (sampled TG: 4,225).
AR PP loses 10.7% from d0 to d128K. Near-flat throughput across that range
remains TODO. Sparse attention and selection are the main depth-dependent
costs. Fresh-load performance variation remains under investigation.

```sh
./result/bin/gufo bench --model "$MODEL" -p 2048 -n 128 \
  -d 0,4096 --temperature 0 --seed 1 -v
./result/bin/gufo bench --model "$MODEL" -p 2048 -n 128 \
  -d 0,4096 --temperature 0 --seed 1 -v \
  --speculative mtp --mtp-model "$MTP"
```

The model uses 2048-token internal chunks, selected by a 512–4096 sweep.
Resident readers gather n-gram rows while layer 0 runs, including graph replay.
HTTP accepts arbitrary prompt lengths and yields between those chunks;
spare session slots do not reduce the idle chunk size.
The [projection sweep](tools/projection_plans.hip) uses `tools/bench/build.sh`.
Projection plans and sparse selections are deterministic. Sparse attention
packs four queries' twelve heads into 48 rows; selection refines a bounded
candidate list with exact score ordering and lowest-index ties.
For a stage breakdown, run `tools/prof/prof.py run --stages qwen-flash --`
before the release command inside `nix develop`. Profiler timings include
instrumentation overhead and are not benchmark results.

`prompt`, `chat` and `serve llm` share MTP and sampling options. Drafts are
greedy over the full vocabulary on the GPU; verification uses the target
sampler with its filters, penalties and RNG. Adaptive MTP is the default;
`--draft-tokens` caps the chain at 1–7 drafts. The controller estimates
accepted output per unit of work, counts only the first rejected proposal
as a failure, and resets with the session. Its decisions depend on acceptance
history, never wall-clock timings.

## Quality checks

Tests and probes live in `tests/models/qwen38_flash_next/`. Build and run the
operator affected by a change first; for example:

```sh
nix develop -c cmake --build --preset gpu-test --target qwen38_flash_next_moe_ids_ops_test
nix develop -c ctest --test-dir build/gpu-test -R 'qwen38_flash_next[.]moe_ids_ops' \
  --no-tests=error --output-on-failure
```

- **Operators:** independent numerical references, FP64 projection dots,
  exact output comparisons and replay. Cover ragged shapes, guards, inactive
  experts, finite outputs, tied selections, and attention near 128K. Router tests check softmax weights and
  lowest-index ties independently of assignment maps.
  Fusions must preserve the separate operators' rounding. Decode and
  verification projections must agree across batch widths one through eight;
  routed down projections also check expanded top-10 batches.
- **Model replay:** compare AR/MTP tokens, full logits, RNG and positions at
  short and 4K contexts, including fresh loads, request order, draft widths
  and resets. Run `build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_session_test
  --model "$MODEL" --mtp-model "$MTP"` for retained model changes.
- **Sampling and wiring:** the shared 23-case serving matrix covers AR/MTP,
  filters, penalties, replay, C2 and short output budgets. Verifier tests cover
  accepted/rejected prefixes, stops and RNG rollback. Wiring changes also
  require actual prompt, chat and sampled HTTP requests.
- **Loading and I/O:** verify upload bytes, guards, shard/chunk boundaries,
  asynchronous n-gram reads, duplicates, failed I/O and teardown.

Prefill changes also require `gufo bench --validate-prefill N` and CPU/reference
probes. Benchmark retained changes with the Nix release and the same artifact,
context capacity and request setup. Keep exact output and acceptance checks
beside speed results; a profiler trace is not a speed benchmark.

Batched prefill differs numerically from sequential decoding. AR/MTP agreement
does not establish unquantized upstream equivalence. Full-model upstream
quality qualification remains TODO.
