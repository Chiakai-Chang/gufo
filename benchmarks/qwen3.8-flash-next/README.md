# Qwen3.8-Flash-Next on Strix Halo

Linux x86-64, AMD `gfx1151`, 128 GB unified memory. Target:
`unsloth/Qwen3.8-Flash-Next-GGUF` at revision
`38bb39ee97821de2c9009abb7e93950eec396e66`, `UD-Q4_K_XL` (four shards).
MTP uses `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` from the same revision.

## Single user

Nix release, pp2048 and tg128, one measured repetition, seed 1,
2026-09-18 UTC. Rates are tok/s. Depth precedes the measured operation;
TG starts from 16 prompt tokens at d0. The CLI uses deterministic repetitive
text, warms prefill and continues generation past EOS. Known greedy rows
use the same 34,817-token context capacity. MTP PP includes draft catch-up.

| Depth | AR pp2048 | MTP pp2048 | AR tg128 | MTP tg128 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1571.61 | 1560.66 | 26.24 | 48.10 |
| 4096 | TODO | TODO | TODO | TODO |
| 8192 | TODO | TODO | TODO | TODO |
| 12288 | TODO | TODO | TODO | TODO |
| 16384 | TODO | TODO | TODO | TODO |
| 32768 | 1423.63 | 1372.12 | 24.04 | 59.12 |
| 65536 | TODO | TODO | TODO | TODO |
| 131072 | TODO | TODO | TODO | TODO |

Separate sampled TG, d0, context 4096:

| Sampling | MTP tg128 | Acceptance |
| --- | ---: | ---: |
| Temperature 0.7, seed 1 | 47.88 | 75.9% |
| Temperature 1.0, top-p 0.95 | TODO | TODO |

The targets of 1700 tok/s pp2048 and near-flat PP through d128K remain TODO.

```sh
./result/bin/gufo bench --model "$MODEL" -p 2048 -n 128 \
  -d 0,32768 --temperature 0 --seed 1 -v
./result/bin/gufo bench --model "$MODEL" -p 2048 -n 128 \
  -d 0,32768 --temperature 0 --seed 1 -v \
  --speculative mtp --mtp-model "$MTP"
```

## Concurrent serving

HTTP aggregate output tok/s, context 4096, greedy, up to 128 output tokens,
one warmup and one measured round per workload. Prefix caching is warm.
These are whole-request rates, including scheduling and prompt handling.
The corpus is [speculative-corpus.json](../qwen3.8-27b/speculative-corpus.json):
`repetition_word` uses identical concurrent requests; mixed requests use
`expository_pangram`, `cpp_ring_buffer` and `reasoning_train`.

| Users | AR repetitive | MTP repetitive | AR mixed | MTP mixed |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 26.81 | 83.96 | 25.81 | 46.52 |
| 2 | 43.19 | 92.89 | 38.80 | 54.51 |
| 4 | 62.35 | 96.41 | 54.53 | 61.32 |
| 6 | 71.78 | 96.90 | 65.62 | 63.06 |
| 8 | 77.52 | 97.88 | 71.74 | 63.46 |

All 46 measured MTP requests matched the AR C1 completions. Acceptance was
100% on repetition and 70.7–71.6% on mixed requests. AR was faster on the mixed
workload at C6/C8.

Target projections share batches across requests; KV, recurrent state,
sampling history and rollback stay independent. MTP proposals still run per
session. Large Q8 projections share weight rows across waves, with eight
input rows per wave. Other shapes retain their measured faster geometry.
`gufo bench` is C1; use `tools/serving/gufo-serving-bench.py` for concurrency.

## Memory and execution

Target weights occupy about 77 GiB of GPU memory. The 26.8 GiB n-gram table
stays on disk; about 6 MiB of compressed rows are cached in RAM. NVMe loading
takes about 15 seconds. Upload staging is released before inference.

C1, context capacity 262144, a 3891-token repetitive prompt and 128 generated
tokens. GPU-visible unified allocations below use the Linux GTT counter with
idle usage subtracted; separate CPU memory is excluded.

| Mode | Loaded GiB | After generation GiB |
| --- | ---: | ---: |
| AR | 85.39 | 85.66 |
| MTP | 89.72 | 90.01 |

KV retains the full requested capacity. Raw indexer keys use a 4096-row ring;
completed blocks keep pooled keys. Transient PLE, MTP and trunk buffers share
storage. Concurrent logit scratch is allocated on first use and sized for the
configured verification width; C1 needs none.

Internal prefill chunks use 2048 tokens. HTTP accepts arbitrary prompt lengths
and yields between chunks. PLE reads and routing-count downloads overlap GPU
work. Prefill fuses SSM convolution and attention preparation into projections.
Headless MTP catch-up retains all KV rows and computes only the final attention
query tile, preserving the carried hidden state.

`prompt`, `chat` and `serve llm` share MTP and sampling options. Adaptive MTP is
the default; `--draft-tokens` caps it at 1–7 proposals. Acceptance history drives
the controller, independent of timing. Snapshots preserve it; new HTTP requests
reset it when reusing context.

A private 0.33 GiB Q4 head shortlists 256 vocabulary rows; the original Q8 head
rescores them. Greedy takes the best rescored row. Sampling draws from the best
64 using the request's filters and penalties. Full-vocabulary target verification
accepts with `min(1, p/q)` or samples normalized `max(p-q, 0)`; the residual draw
becomes the next anchor. Draft shortlisting does not truncate the target.

Experiments not retained: grouped Q5 down projections were slower; adaptive
two/four-query sparse attention gave no clear model-level gain. Larger attention
tiles and wider DeltaNet reductions were also slower.

## Maintaining quality

Tests and probes live in `tests/models/qwen38_flash_next/`. The
`qwen38_flash_next_tests` target contains 13 distinct operator/I/O checks.
Build and run only the affected checks during an experiment, for example:

```sh
nix develop -c cmake --build --preset gpu-test --target qwen38_flash_next_projection_ops_test
nix develop -c ctest --test-dir build/gpu-test -R 'qwen38_flash_next[.]projection_ops' \
  --no-tests=error --output-on-failure
```

- **Operators:** independent numerical references, FP64 dots, output guards,
  ragged shapes, finite values, tied selections and replay. Q8 projections must
  match scalar evaluation at widths 1–8 and 16/24/32. Grouped experts cover
  duplicate and inactive slots. Attention checks include deep contexts and
  exact agreement between full and final-tile catch-up. Indexer masks match a
  CPU full sort. N-gram reads cover cold/cache/mixed paths and I/O failures.
- **Model state:** `qwen38_flash_next_session_test` checks full logits, tokens,
  RNG and acceptance against serial execution at C2/C4/C6/C8, including ragged
  budgets, reordered requests, a 4K boundary and complete snapshot bytes.
  Its shared 23-case serving matrix checks AR/MTP filters, penalties, replay,
  C2 and short budgets. Sampled tokens teacher-forced through AR must reproduce
  each frontier. `qwen38_flash_next_snapshot_test` checks persistence, rollback
  and deferred residual draws. Both accept `--model "$MODEL" --mtp-model "$MTP"`
  and belong to the explicit `qwen38_flash_next_model_tests` build target.
- **Qualification:** scheduling/fusion changes must preserve operator rounding
  and model replay. Arithmetic changes additionally need CPU/reference probes
  and `gufo bench --validate-prefill N`. The current 1024-token prefill check has
  finite logits, scalar-winner rank 1, RMSE 0.24 and maximum error 1.15. Snapshot
  compatibility changes when inference arithmetic changes.

Measure retained changes with Nix release binaries and the same artifact,
capacity, prompts and sampling setup. Keep output/acceptance checks beside
rates. Profiler timings are diagnostic, not benchmark results.
`tools/prof/prof.py run --stages qwen-flash -- ...` profiles a release command;
[projection_plans.hip](tools/projection_plans.hip) uses `tools/bench/build.sh`.
Standalone ablations must match production flags:
`-O3 -ffast-math -fno-finite-math-only`.

Sampled MTP preserves the target distribution within floating-point precision;
it need not match AR's same-seed tokens or prefixes across different budgets.
Greedy decoding retains those guarantees. Batched prefill differs numerically
from sequential decoding. These regression checks do not establish equivalence
to the unquantized upstream model; that qualification remains TODO.
