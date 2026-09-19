# Qwen3.8-Flash-Next on Strix Halo

Linux x86-64, AMD `gfx1151`, 128 GB unified memory. Nix release, measured
2026-09-19, one repetition per point. Target: `unsloth/Qwen3.8-Flash-Next-GGUF`
revision `38bb39ee97821de2c9009abb7e93950eec396e66`, `UD-Q4_K_XL` (four shards).
MTP: `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` from the same revision.

Chat follows the official template: thinking on, `xhigh` effort, prior reasoning
preserved. PNG/JPEG input works with AR/MTP and `mmproj-BF16.gguf`; see
[image usage and validation](../qwen3.8-27b/eval/vision.md).
**Original unquantized-model and GGUF-conversion parity remain unqualified.**

## Single user

Tok/s, pp2048/tg128, greedy, seed 1, context capacity 133121. Depth precedes
the timed operation. The raw CLI uses a fixed repeated token pattern, warms
prefill and continues past EOS; d0 TG starts from 16 tokens. MTP PP includes
predictor catch-up through all known successor tokens.

| Depth | AR pp2048 | MTP pp2048 | AR tg128 | MTP tg128 | MTP acceptance |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1525.9 | 1473.2 | 26.33 | 42.74 | 70.0% |
| 4K | 1455.1 | 1399.4 | 25.48 | 40.35 | 65.6% |
| 8K | 1432.0 | 1374.4 | 25.25 | 35.68 | 58.3% |
| 12K | 1421.4 | 1362.2 | 25.19 | 52.03 | 85.7% |
| 16K | 1412.3 | 1352.8 | 24.91 | 56.31 | 88.5% |
| 32K | 1390.1 | 1328.2 | 24.10 | 61.59 | 100.0% |
| 64K | 1342.7 | 1282.1 | 23.08 | 59.67 | 100.0% |
| 128K | 1282.2 | 1225.4 | 22.14 | 57.65 | 100.0% |

AR prefill falls 16.0% from d0 to d128K. The 1700 tok/s target and flat
deep-context throughput remain unmet.

```sh
./result/bin/gufo bench --model "$MODEL" -p 2048 -n 128 \
  -d 0,4096,8192,12288,16384,32768,65536,131072 --temperature 0 --seed 1 -v
# Add --speculative mtp --mtp-model "$MTP" for MTP.
```

Sampled MTP, d0 raw prefix, seed 1, tg128, top-k/top-p/min-p disabled:

| Sampling | MTP tok/s | Acceptance |
| --- | ---: | ---: |
| T=0.7 | 43.41 | 65.2% |
| T=1.0 | 29.16 | 56.5% |

## Concurrent serving

Aggregate **output** tok/s, including prompt handling and scheduling. Context
4096, greedy, thinking off, 128 output tokens, one cold cohort on a fresh server
per point; no cache hits. The [corpus](../qwen3.8-27b/speculative-corpus.json)
uses `repetition_word` for repetition and distinct requests cycling through
`expository_pangram`, `cpp_ring_buffer`, `reasoning_train` for mixed work.

| Users | AR repetitive | MTP repetitive | AR mixed | MTP mixed |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 25.24 | 62.56 | 25.12 | 41.92 |
| 2 | 38.60 | 77.14 | 36.25 | 52.94 |
| 4 | 53.09 | 87.00 | 48.41 | 58.80 |
| 6 | 60.68 | 89.88 | 54.51 | 63.16 |
| 8 | 65.17 | 91.41 | 58.45 | 67.74 |

MTP acceptance is 100% on repetition and 72.9–85.3% on the mixed cohorts.
Every completion matches AR; all cohorts report zero cache hits.

**C1 is a single user.** The HTTP and raw CLI prompts above differ. With the
same repetitive chat prompt, CLI decode measured 75.42 tok/s and HTTP
C1 75.85 tok/s. Whole-request HTTP throughput also includes prefill
and scheduling. Compare identical prompts and timing boundaries.
`gufo bench` is C1; use `tools/serving/gufo-serving-bench.py` for concurrency.

## MTP and profiling

Adaptive MTP is the default; `--draft-tokens` caps it at 1–7 proposals.
Sampled requests use deterministic per-session acceptance history and fixed
cost curves, preserving seeded replay independently of scheduler timing.
The curves cover C1/C2/C4/C6/C8 and d0/d4K/d32K; they interpolate context costs
and extrapolate the measured slope to the native 262144-token limit.

All-greedy C>1 batches choose one width from live cycle timings and conditional
acceptance at each depth. Occupancy/context bins remain separate. Plain controls
start after occupancy stabilizes; width transitions are excluded from estimates
because they replay the preceding chain. C1 uses deterministic acceptance-based
control.

Predictor QKV/output, expert and vocabulary projections batch across requests;
KV, positions, recurrent state, rollback and RNG remain private. MTP uses
ratio-four QSA with independent index caches and snapshot/rewind state. Ranking
retains FP32 queries, FP16 pooled keys and a fixed FP32 reduction. Verification
uses `min(1, p/q)` acceptance and normalized `max(p-q, 0)` residual correction.
Temperature precedes top-p/min-p for both target and draft sampling.

The d32K pp2048 trace spends 29.1% of kernel time in MoE, 34.9% in dense
projections and 12.6% in attention/indexing. The C4 mixed MTP trace (32 output
tokens/request) is 80.7% GPU-busy across its inference span; MoE/MMQ accounts
for 51.3% and dense projections 36.1% of kernel time. Throughput tables use
separate unprofiled runs. Packing more vocabulary rows per Q8 block gave no
C1 improvement; that experiment was not retained.

MTP normalizes the complete 10240-wide hidden stream. Separate embedding and
hidden projections avoid repeating the embedding work across four HC branches.
The original Q8 head scores the full vocabulary; sampled proposals use its exact
top 64 with p/q verification. There is no private Q4 head. Predictor inputs use
shifted text-token embeddings, with image information carried by target hidden
states and mRoPE. See the [source audit and independent oracle](eval/mtp.md).

## Continuations and memory

An immutable prompt snapshot supports branching; the final live session serves
likely continuations without re-prefilling executed assistant tokens. First-token
publication precedes capture on a worker; other sessions keep decoding while
that session is frozen. Disk-only capture reserves staging capacity before
allocation. A bounded worker streams/checksums writes outside the lookup gate.
GGUF compatibility uses cached full-content SHA-256.

MTP trails prefill by one token; kept hidden state is 320 KiB/session. Rollback
rows allocate progressively, up to about 788 MiB for seven drafts. Restoration
retains only existing rows useful to the restored policy and remaining context;
reset releases all rows. A 4095-token MTP snapshot occupies 221 MiB. Session
admission accounts for context state, the configured rollback cap and shared
scratch before allocation.

MTP projection scratch reuses target scratch; no separate concatenation buffer
or output-head copy is allocated.

Image snapshots require a matching prompt attachment; text restores clear image
state. Cancellation covers prefill layers, vision tiles and snapshot regions.
Failed mutations invalidate the session until reset/rebuild.

## Maintaining quality

All eight greedy depth points and all fresh-server cohorts match AR. The
2176-token scalar/prefill control retains the same top-1 token (logit RMSE 0.18).

Tests/probes live in `tests/models/qwen38_flash_next/`. The
`qwen38_flash_next_tests` target contains 14 focused operator/configuration/I/O
checks. Build only affected targets during iteration:

```sh
nix develop -c cmake --build --preset gpu-test --target qwen38_flash_next_select_ops_test
nix develop -c ctest --test-dir build/gpu-test -R 'qwen38_flash_next[.]select_ops' \
  --no-tests=error --output-on-failure
```

- **Operators and loading:** independent scalar/FP64 references, exact FP32
  selector scores, near-tie ranking cases, ragged shapes, guards and replay.
  Malformed compression/mRoPE metadata, unsupported kernel geometry and
  incompatible MTP sidecars fail loading.
- **State and sampling:** `qwen38_flash_next_session_test --batch-only` compares
  tokens/full logits and sampled RNG/acceptance against isolated execution at
  C2/C4/C6/C8, including ragged budgets, reordered requests, bounded rollback,
  cancellation recovery and image attachments. `--sampling-only` covers 23
  AR/MTP configurations, penalties, residual correction and short budgets.
  `qwen38_flash_next_snapshot_test` checks persistence and continuation replay.
  AR sessions sharing an MTP-capable model allocate no predictor state;
  mixed-mode target logits match exactly and snapshots cannot cross modes.
- **Prefill and serving:** `--prefill-only` checks full logits across boundaries
  through 4096 tokens, including 1/8/9/32/33-token tails. Image checks cover
  AR/MTP, concurrency, RAM reuse and disk restoration. Official template and
  Unicode/NFC token goldens cover both Qwen models. HTTP tests require complete
  UTF-8 in every streamed JSON event and schema-correct tool arguments, including
  quoted closing markers and calls after unclosed reasoning.
- **Audits:** `qwen38_flash_next_gpu_probe --mtp-audit` checks original encoded
  weights, full-width normalization, predictor stages with independently
  computed caches, recursive carry and independent text/image batches against
  a scalar CPU oracle. `--cost-audit C`
  measures warmed catch-up/proposal/verification costs (`0` selects
  C1/C2/C4/C6/C8). Both avoid storing logit fixtures.

The model tests/probes accept `--model "$MODEL" --mtp-model "$MTP"`; build the
session/snapshot tests with `qwen38_flash_next_model_tests`. Preserve arithmetic
and replay when optimizing layout/fusion. Arithmetic corrections additionally
need independent references and `gufo bench --validate-prefill N`. Measure with
Nix release binaries, matching artifacts, capacity, prompts and sampling;
profile separately using `tools/prof/prof.py run --stages qwen-flash -- ...`.

Sampled MTP preserves the target distribution within floating-point precision.
Replay requires the same seed, request budget, configured capacity and sampling
settings; sampled MTP need not match AR's same-seed sequence. Greedy output must
remain independent of draft width and batching.

The pinned official image processor allows 64–16384 merged tokens; a 1024-token
cap changes resolution and output. Short prefill tails retain the arithmetic of
larger chunks. The artifact uses native RoPE without YaRN extension. Official
Transformers `c587bc884db2c2e31fc2b8102314656b17aa07b1` defines FP32 QSA but ignores
MTP weights; operator/export audits do not close original-model parity.
