# Qwen3-ASR benchmarks

Linux gfx1151, BF16, Nix release. RTF = request wall time / input audio duration
(lower is better). Model loading is excluded from resident request timings.

## Single request

Latest retained route audit (September 2026): 15.051229-second English fixture,
49 greedy output tokens matching the official reference.

| Metric | Result |
| --- | ---: |
| Resident request mean | 992.14 ms |
| RTF | 0.0659 |
| Audio per wall second | 15.17 |
| Load, warm filesystem | 389.198 ms |

Earlier separate stage/profile control: resample 6.27 ms, log-mel 7.18 ms,
audio encoder 36.02 ms, text prefill/decode 942.44 ms. Decode projections account
for 77.8% of GPU time, sustaining 91–99% of the measured 240.24 GB/s read ceiling.
Do not sum stages from different runs into a new headline.

## Concurrent HTTP

2026-09-10, Base audio fixture 10.24 s, greedy; synchronized requests. These
numbers predate the latest loading cleanup and require a current refresh.

| C | Audio-s / wall-s | Latency p50 | Latency max |
| ---: | ---: | ---: | ---: |
| 1 | 15.71 | 0.65 s | 0.65 s |
| 2 | 15.73 | 0.98 s | 1.30 s |
| 4 | 15.73 | 1.63 s | 2.60 s |
| 6 | TODO | TODO | TODO |
| 8 | 15.74 | 2.93 s | 5.20 s |

Work is serialized. More connections increase queue latency, not model throughput.
The matching audio.cpp `3174e6b` BF16 comparison averaged RTF 0.121 versus Gufo
0.075 on six clips; all six transcripts matched. This is a historical comparison,
not a general ASR accuracy score.

## Reproduce

```sh
nix build
nix develop -c python3 src/models/qwen3_asr/tools/benchmark.py \
  --model "$MODEL" --audio speech.wav
```

Use one resident runtime, one warmup and five measured requests; report the
median and exact fixture/engine/model identities. Current long-form corpus,
streaming and C6 measurements: **TODO**. Raw output stays outside Git.
