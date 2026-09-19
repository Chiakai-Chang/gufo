# DSpark sampling qualification

Issue #238, 2026-09-19. Same target/support artifacts and gfx1151 hardware as
the [benchmark README](README.md). This evaluates proposal and verification
semantics; it does not establish official-checkpoint quality for the quantized
target. The existing target-reference alerts remain in [eval/README.md](eval/README.md).

## Default behavior

| Request | Proposal | Compatibility |
| --- | --- | --- |
| Greedy | Original Markov argmax | Original tokens and dispatch path |
| C1 throughout the request, including sampled prompt/chat | Point mass | Same sampled AR trace for matching target logits and seed |
| C>1 with top-p <1 or top-k 2–256 | Hybrid top-8 / point mass | Exact target distribution; its own RNG trace |
| Unfiltered or otherwise expensive sampled distributions | Point mass | Same sampled AR trace for matching target logits and seed |

`top-k=1` is deterministic unless `min-keep` retains more candidates.
Hybrid selection applies temperature to eight Markov-corrected candidates.
When the largest probability is at least 0.7, it uses a point mass at that
candidate; otherwise it draws from the shortlist. This decision precedes the
draw. Shortlist masses are multiples of 2^-24, sum exactly to one in F32, and
match the proposal RNG's resolution.

For a proposed token `v`, acceptance is `min(1, p(v)/q(v))`. Rejection samples
the full target vocabulary from `normalize(max(p-q,0))`, including tokens
outside the shortlist. Point-mass verification draws once from `p`, accepting
only a matching token. Both routes preserve `p`: accepted mass is
`min(p(v),q(v))` and unconditional residual mass is `max(p(v)-q(v),0)`.
Precision is bounded by the existing floating-point arithmetic and RNG.

All target filters and penalties apply to complete target logits and committed
history. Each request owns its RNG, proposals and acceptance state. A rejected
draw becomes the next cycle's anchor; discarded suffixes do not enter history.
Prompt, chat, bench and HTTP use the same bridge. Bench includes prompt tokens
in penalty history; each new chat turn clears pending draws and controller
history while preserving valid prefix state. AR chat commits its complete
emitted prefix before reuse, matching the server and DSpark cache boundary.

Seed replay requires identical inputs, sampling settings and dispatch schedule.
Changing concurrency, cohort membership or the proposal policy may change a
sampled trace. There is no production policy switch. C1 retains the qualified
point-mass path because stochastic proposal overhead did not repay its cost.

## Official formulas

The source audit pins:

- DeepSpec `005e03b81cec38b7da6399833d609ee89a2587f2`:
  `deepspec/modeling/dspark/markov_head.py`,
  `deepspec/eval/base_evaluator.py`, and
  `deepspec/eval/dspark/draft_ops.py`.
- `deepseek-ai/DeepSeek-V4-Flash-0731`
  `7872f01b1d1fe23eabc4c98b48bffcef5a386062`, `inference/model.py`:
  `DSparkBlock.forward_head` and `DSparkConfidenceHead`.

The Markov correction is `base_logits + W2(W1(previous_token))`.
DS4 confidence is
`sigmoid(projection(concat(hc_head(hidden), W1(previous_token))))`, with no
bias. Confidence uses the **pre-final-RMSNorm** hidden state; the LM head uses
the normalized state. The generic DeepSpec Qwen/Gemma feature path is not the
DS4 confidence formula.

The GPU fuses Markov correction with partial top-8 selection and returns eight
IDs/logits plus confidence per request. Additional selection scratch is lazy
and about 256 KiB for eight requests; no additional dense corrected-logit or
probability cache is allocated. Existing support-weight caching includes the
confidence tensor.

Confidence predicts prefix survival. The scheduler combines this estimate with
the existing offline gfx1151 cycle-cost table and observed acceptance. It
decides whether to include the current position **before sampling it**, never
retracts earlier proposals, and never uses live timing to make token decisions.
An unprofitable current block falls back to target decode and retries after
one draft budget. Historical rejection retains the existing longer backoff.

## Dense, sparse and confidence controls

An isolated diagnostic captured three short prompts (club names, autumn leaves,
and a repeating color pattern), at temperatures 0.6 and 1.0, top-p 0.95, seed 7.
Confidence stopping was disabled in that diagnostic build. Independent CPU
Markov evaluation checked GPU candidates against the actual quantized weights.
The table compares proposals on the **same 126 verified conditional rows**.

| Temperature | Rows | Point mass expected acceptance | Top-4 | Top-8 | Dense q |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6 | 63 | 0.6902 | 0.6990 | 0.6992 | 0.6988 |
| 1.0 | 63 | 0.6295 | 0.6673 | 0.6698 | 0.6601 |

Expected stochastic acceptance is `sum(min(p,q))`. The CPU dense-distribution
control took 3.257 s over 183 proposed rows, versus 0.415 ms for the sparse
distribution construction. This excludes CPU Markov calculation/capture and
is **not** a comparison against an optimized dense GPU implementation.

For top-8 peak probability below 0.7, stochastic proposals improved expected
acceptance in this sample. Above 0.7, point masses performed better. This
motivates the hybrid boundary; it is a hardware/workload decision, not a
quality approximation.

| Predicted confidence | Rows | Mean prediction | Observed acceptance |
| --- | ---: | ---: | ---: |
| [0, 0.4) | 30 | 0.273 | 0.267 |
| [0.4, 0.6) | 33 | 0.492 | 0.333 |
| [0.6, 0.8) | 16 | 0.703 | 0.938 |
| [0.8, 1] | 47 | 0.945 | 0.894 |

This is a small, correlated calibration sample. The confidence head separates
weak and strong positions, but is not perfectly calibrated. No remapping was
fitted to these rows. Confidence affects work scheduling, never the rejection
formula. The original release supplies the point-mass/no-confidence control;
temporary diagnostic builds supply ablations without production switches.

## Retained controls

The identical-input C1 HTTP control produced the same 128-token completion
across the original point-mass release, the hybrid release, AR, and cold/cached
repeats (six outputs; SHA-256
`e38e5f3c9f7eb06eb743bbe396200e5900eb09df201cc6c99a368135ea492df9`).
Point-mass DSpark measured 18.99–19.01 tok/s in those four runs.

The [retained artifact](dspark-sampling.json) is also the fixed input corpus.
Two fresh-server C2 measurements in opposite variant order, temperature 1,
top-p 0.95, seed 7, context capacity 8192, four server slots, no cache hits:

| Prompt | Hybrid per-user decode tok/s | Change versus point-mass control | Hybrid accepted/drafted |
| --- | ---: | ---: | ---: |
| Autumn explanation | 17.40 | +10.2% | 92/140 |
| Repeating color pattern | 21.01 | +21.9% | 168/204 |
| Six club names | 15.27 | -3.1% | 4/4 |

Aggregate end-to-end output throughput over the three cases was **29.91 tok/s**,
11.9% above the point-mass control. Output hashes and draft counts repeated for
every same-mode C2 request. The naming task ended after 23 tokens for the
control and 21 for hybrid; it is a short-latency limit, not a fixed tg128 run.
The other cases emitted 128 tokens per request.

C4 had a small positive single-screen result with warm prefix reuse. It is not
a repeated cold qualification. Broader temperature/depth/seed speed sweeps
remain TODO; no universal speedup is claimed.

To reproduce one cold C2 run, start a fresh server with the same release:

```sh
./result/bin/gufo serve llm --model "$MODEL" --dspark-model "$DSPARK" \
  --speculative dspark --served-model-name ds4 --context 8192 \
  --sessions 4 --seed 7 --top-p 0.95
nix develop -c tools/serving/gufo-serving-bench.py --model ds4 \
  --suite benchmarks/deepseek-v4-flash/dspark-sampling.json \
  --corpus-layout homogeneous --concurrency 2 --max-tokens 128 \
  --temperature 1 --warmup 0 --repetitions 1 --output /tmp/ds4-sampled.json
```

Restart the server for the repeat. Use fixed `--suite` inputs: ordinary
`--prompt` benchmarks insert a random nonce, invalidating identical-input
hash/acceptance comparisons across invocations.

## Quality gates

The Nix release build and the following focused checks passed. Greedy replay
matched all **736 tokens** and every compared full-logit vector exactly.

- `ds4.sampling`: 840,000 Monte Carlo trials, including conditional two-position
  proposals, mixed delta/stochastic rows, residual mass outside q, committed
  penalties, RNG replay, malformed inputs and predictable confidence stopping.
- `logit_sampler_test`: all maintained sampling configurations, comparing the
  bounded target-distribution selector with the full-sort reference on both
  concentrated and diffuse 4,099-token rows; probability tolerance 1e-12.
  Existing seeded AR draw checks remain unchanged.
- `ds4.attention`: independent double-precision Markov and pre-norm confidence
  formulas at C1/2/3/4/6/8, nontrivial strides and vocabulary tails, alongside
  existing attention/indexer gates.
- `ds4.serving`: C1 AR/DSpark seeded equality, deterministic sampled batch replay,
  mixed greedy/sampled cohorts, prefix/disk restore, late prefill and exhaustion.
- `ds4.chat`: real standalone prompt and two sampled chat turns, including
  truncation at the generation budget, with identical AR/DSpark token traces.
- `ds4_quality_test --dspark-replay`: retained exact greedy tokens/frontier
  logits through C8/16K, including snapshots and changing concurrency.

Use the existing DS4 tests and shared serving/profiling tools. No separate
production reference route or duplicate model test suite is needed.
