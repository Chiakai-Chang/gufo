# DeepSeek V4 Flash on Strix Halo

Linux x86-64, AMD `gfx1151`, 128 GB unified memory. Nix release binaries.
`C` is simultaneous requests. Single-user measurements use **pp2048 / tg128**;
depth precedes the measured operation. Unknown current measurements are **TODO**.

Target: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
(80.76 GiB), `antirez/deepseek-v4-gguf` revision
`1cd7b564460821938add0475a60b942c409295e0`. DSpark support:
`DeepSeek-V4-Flash-DSpark-support-0731.gguf`, revision
`e7f04037032990db0346398d249baf9fb9df1ccc`.

## Single user, autoregressive

Latest depth control: **2026-09-19**, two release measurements at d0/d32K.
Values are mean ± standard deviation; initial kernel setup contributes to
depth-zero prefill variation. The final packing layout needs a d64K speed refresh.

| Context depth | pp2048 tok/s | tg128 tok/s |
| ---: | ---: | ---: |
| 0 | 450.26 ± 33.18 | 17.74 ± 0.00 |
| 4,096 | TODO | TODO |
| 8,192 | TODO | TODO |
| 12,288 | TODO | TODO |
| 16,384 | TODO | TODO |
| 32,768 | 422.18 ± 3.61 | 14.86 ± 0.00 |
| 65,536 | TODO | TODO |

## Single user, DSpark

Same workload and measurement date: two measurements at d0, one at d32K.
DSpark prefill includes support-state work.

| Context depth | pp2048 tok/s | tg128 tok/s |
| ---: | ---: | ---: |
| 0 | 453.49 ± 25.96 | 17.45 ± 0.00 |
| 4,096 | TODO | TODO |
| 8,192 | TODO | TODO |
| 12,288 | TODO | TODO |
| 16,384 | TODO | TODO |
| 32,768 | 417.80 | 35.30 |
| 65,536 | TODO | TODO |

The CLI uses a repeating token sequence. Natural prompts can have substantially
different acceptance and speed. Depth-zero generation starts from 16 tokens;
it does not follow the measured 2048-token prefill.

## Multiple users, autoregressive

Current pp2048/tg128 depth sweep: **TODO**. Report aggregate prompt throughput
and per-user generation throughput at each depth.

| Users | d0 / d4K / d8K / d12K / d16K / d32K |
| ---: | --- |
| 2 | TODO |
| 4 | TODO |
| 6 | TODO |
| 8 | TODO |

## Multiple users, DSpark

Current pp2048/tg128 depth sweep: **TODO**, including acceptance and output
checks at every concurrency.

| Users | d0 / d4K / d8K / d12K / d16K / d32K |
| ---: | --- |
| 2 | TODO |
| 4 | TODO |
| 6 | TODO |
| 8 | TODO |

Earlier raw [greedy](speed-matrix.json) and [sampled](sampled-speed-matrix.json)
matrices retain their release identities, deviations, hashes and counters.
They are not a refresh of the current kernels.

Latest fixed-prompt sampled HTTP control (2026-09-19, C2, temperature 1,
top-p 0.95, seed 7): **17.40 tok/s per user** for an autumn explanation,
**21.01** for a repeating pattern, and **15.27** for a short naming task.
Two fresh-server measurements; details and limits are in the
[DSpark sampling report](dspark-sampling.md).

DSpark combines offline cycle costs and acceptance history with checkpoint
confidence for filtered sampled C>1 requests. These use exact hybrid top-8
proposals; C1 and unfiltered sampling retain point-mass proposals. Stochastic
verification preserves the target distribution, with its own seeded trace.
Greedy behavior is unchanged. Requests keep private RNG, proposal, acceptance
and controller state; prefix reuse starts fresh request statistics. Live timing
never changes token decisions.

## Memory

C1 with context capacity **262,144**. GPU-visible unified allocations exclude
separate CPU memory; capacity does not imply filling the window.

| DSpark workload | GPU allocation |
| --- | ---: |
| pp2048/4096 + tg128 | 90.74 GiB |
| 16K prefix, pp4096 + tg128 | 91.46 GiB |

The [memory control](memory-c1-262k.json) records these measurements. Target and
support weights, KV and working buffers all contribute. Compressed KV and score
scratch grow with actual use. Prefill reuses scratch across ordered stages;
indexer scoring borrows the idle attention-output range for temporary F16 keys.
Queries share that range and feed WMMA directly. Persistent KV precision is
unchanged.

## Reproduce

```sh
nix build
MODEL=/path/to/target.gguf
DSPARK=/path/to/DSpark-support.gguf
./result/bin/gufo bench --model "$MODEL" \
  -c 1 -p 2048 -n 128 -d 0,32768 -r 2 -v
./result/bin/gufo bench --model "$MODEL" --dspark-model "$DSPARK" \
  -c 1 -p 2048 -n 128 -d 0,32768 -r 2 -v
```

Use `-c 1,2,4,6,8` for concurrency and
`-d 0,4096,8192,12288,16384,32768` for the full depth sweep. Use `-p 4096`
for pp4096. Sampled controls use the same `--temperature` and `--seed` in both
modes. Context preparation and restoration are outside timing. Verbose output
records hashes and draft counts; compare these alongside speed. Run model
jobs, builds and profiles sequentially. The HTTP benchmark is
`tools/serving/gufo-serving-bench.py`; distinguish cold requests from cache hits.

## Maintain quality

Checks live in `tests/models/deepseek_v4_flash`; maintained commands live in
[tools/ds4](../../tools/ds4/README.md). Run only affected checks while iterating.

| Check | Required coverage |
| --- | --- |
| `ds4.template`, `ds4.cli`, `ds4.dataset`, `ds4.eval` | Framing, wiring, pinned fixtures and probability/grading invariants |
| `ds4.sampling` | Exact p/q and residual distributions, conditional proposals, penalties, seed replay and confidence stopping |
| `ds4.projections` | Independent weight decoding, FP64 formulas, all IQ2 signs, MoE ownership, exact batch/scalar outputs, official HC/Sinkhorn equations |
| `ds4.attention` | Independent attention, DSpark Markov/confidence and window formulas; exact prefill scores, masks, poisoned rows, ties, scratch bounds and top-k ordering |
| `ds4.target` | Official tokens, pinned trajectory, full logits, replay, capacity equality and state isolation |
| `ds4.dspark` | Exact scalar tokens/frontier logits through C8/16K, acceptance, policy, snapshots and forks |
| `ds4.serving` | Sampling, physical batching, EOS checkpoint isolation, bounded prefill, prefix/disk caches, cancellation and exhaustion |
| `ds4.chat` | Real sampled prompt and three-turn chat after length/EOS stops, AR/DSpark token identity and active drafting |

```sh
nix develop -c tools/ds4/check.py fast
nix develop -c tools/ds4/check.py kernels
nix develop -c tools/ds4/check.py model --model "$MODEL" --dspark-model "$DSPARK"
```

Numerical changes require independent operator controls and model qualification.
The pinned trajectory requires **116/128 top-1 choices**, rank sum ≤142 and
worst rank ≤3. State comparisons require finite logits, **RMSE ≤1.12,
cosine ≥0.979, max error ≤5**. DSpark replay covers **736 exact token choices**
through C8/16K. Never loosen these limits to accept a faster implementation.
Repeatability assumes the same inputs, seeds and execution schedule. It does
not establish general equality across different batch compositions.

Four post-prefill differential alerts against Antirez remain unresolved;
Antirez is not official ground truth. AR/DSpark agreement cannot detect shared
model errors. Original-checkpoint distribution parity and full capability
qualification remain TODO. See [quality evidence and limits](eval/README.md).

The latest optimized target check also misses the historical trajectory guard:
115/128, rank sum 145, worst rank 4. The unchanged baseline produces identical
full logits, while the legacy Debug configuration passes at 116/128, 142 and 3.
The limits remain unchanged. Indexer changes match the baseline's ten complete
prefill vectors across 2K/4K calls and five depths; this establishes preservation
on those cases. At 64K, five full-logit vectors also match across disk-snapshot
transfer, continued prefill and decoding. These checks establish preservation
on those cases, not resolution of the existing build-sensitive discrepancy.

## Current implementation notes

- Prefill indexer scoring packs queries once and reuses F16 key fragments across its 64 heads, preserving the existing products and reduction order.
- Exact partial top-k covers wide prefill through 32,768 compressed keys; deeper inputs use the parallel fallback. Tie ordering is unchanged.
- Scalar-equivalent speculative projections and attention retain independent request state.
- Additional F16 rounding in HC and FP32 compressor/router experiments were rejected by the maintained quality controls.
- Paired IQ2 gate/up and transposed sparse-value experiments (2026-09-19) were rejected: the exact variants were slower; smaller paired tiles spilled registers and failed exactness.
