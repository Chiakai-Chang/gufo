# Qwen3.8-Flash-Next on Strix Halo

Linux x86-64, AMD `gfx1151`, 128 GB unified memory. Nix release, measured
2026-09-19, one repetition per point. Target: `unsloth/Qwen3.8-Flash-Next-GGUF`
revision `38bb39ee97821de2c9009abb7e93950eec396e66`, `UD-Q4_K_XL` (four shards).
MTP: `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` from the same revision.

Chat follows the official template: thinking on, `xhigh` effort, prior reasoning
preserved. PNG/JPEG input works with AR/MTP and `mmproj-BF16.gguf`; see
[image usage and validation](README.md#images).
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
| T=0.7 | 39.83 | 65.2% |
| T=1.0 | 28.45 | 56.5% |

## Concurrent serving

Aggregate **output** tok/s, including prompt handling and scheduling. Context
4096, greedy, thinking off, 128 output tokens, one cold cohort on a fresh server
per point; no cache hits. The [corpus](../qwen3.8-27b/artifacts/speculative-corpus.json)
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
