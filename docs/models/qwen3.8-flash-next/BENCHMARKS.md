# Qwen3.8-Flash-Next on Strix Halo

Linux x86-64, AMD `gfx1151`, 128 GB unified memory. Production builds with the
pinned Nix toolchain. PP/TG at d0/d32K/d128K: 2026-09-21; serving and vision:
2026-09-20; other CLI measurements: 2026-09-19. One repetition per point.
Target: `unsloth/Qwen3.8-Flash-Next-GGUF`
revision `38bb39ee97821de2c9009abb7e93950eec396e66`, `UD-Q4_K_XL` (four shards).
MTP: `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` from the same revision.

Chat follows the official template: thinking on, `xhigh` effort, prior reasoning
preserved. PNG/JPEG input works with AR/MTP and `mmproj-BF16.gguf`; see
[image usage and validation](README.md#images).
**Original unquantized-model and GGUF-conversion parity remain unqualified.**

Reference implementation: **llama.cpp** `llama-server` from this repository's
`flake.nix` (ROCm, gfx1151, same sharded GGUF), measured over HTTP with the
same prompts and timed scope. **Gain** is Gufo over llama.cpp, positive when
Gufo is faster; `Gain vs llama.cpp AR` compares MTP against llama.cpp without
a draft. All llama.cpp cells are **TODO** until the first HTTP sweep. Layout
and method: [benchmark-model skill](../../../.agents/skills/benchmark-model/SKILL.md).

## Loading

Cold-file-cache launch to HTTP readiness on 2026-09-21, including all four
target shards, MTP and two sessions at context capacity 262144.

<!-- bench:loading -->
| Target | Gufo ready | llama.cpp ready | Gain |
| --- | ---: | ---: | ---: |
| UD-Q4_K_XL | 14.91 s | TODO | TODO |
<!-- /bench -->

![Cold-file-cache launch to readiness](artifacts/charts/loading.svg)

A direct 4095-token snapshot occupies **221 MiB**; restore/replay
qualification is in [evaluation](EVALUATION.md).

## Single user, autoregressive

Tok/s, pp2048/tg128, greedy, seed 1, context capacity 133121. Depth is a
cached prefix and precedes the timed operation. The raw CLI uses a fixed
repeated token pattern, warms prefill and continues past EOS; d0 TG starts
from 16 tokens. Gufo values from `gufo bench`, d0/d32K/d128K refreshed
2026-09-21, other depths 2026-09-19.

<!-- bench:single-ar -->
| Depth | Gufo pp | llama.cpp pp | Gain | Gufo tg | llama.cpp tg | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1537.5 | TODO | TODO | 26.43 | TODO | TODO |
| 4,096 | 1455.1 | TODO | TODO | 25.48 | TODO | TODO |
| 8,192 | 1432.0 | TODO | TODO | 25.25 | TODO | TODO |
| 12,288 | 1421.4 | TODO | TODO | 25.19 | TODO | TODO |
| 16,384 | 1412.3 | TODO | TODO | 24.91 | TODO | TODO |
| 32,768 | 1406.4 | TODO | TODO | 24.28 | TODO | TODO |
| 65,536 | 1342.7 | TODO | TODO | 23.08 | TODO | TODO |
| 131,072 | 1310.5 | TODO | TODO | 22.33 | TODO | TODO |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

AR prefill falls 14.8% from d0 to d128K. Single runs vary by about 1% within
a session and the machine drifted 2% across one evening; the d0 prefill
change was qualified by interleaved runs of both Nix binaries
(1559.6 → 1568.1 tok/s). The 1700 tok/s target and flat deep-context
throughput remain unmet.

## Single user, MTP

Same workload with `--speculative mtp`. MTP PP includes predictor catch-up
through all known successor tokens and adds about 2% to prefill time at the
three refreshed depths; greedy output matches AR at each depth. llama.cpp has
no MTP for this model: the reference column is llama.cpp autoregressive
generation with the same target GGUF.

<!-- bench:single-mtp -->
| Depth | Gufo pp | Gufo tg | Acceptance | llama.cpp AR tg | Gain vs llama.cpp AR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1503.6 | 43.95 | 69.2% | TODO | TODO |
| 4,096 | TODO | 40.35 | 65.6% | TODO | TODO |
| 8,192 | TODO | 35.68 | 58.3% | TODO | TODO |
| 12,288 | TODO | 52.03 | 85.7% | TODO | TODO |
| 16,384 | TODO | 56.31 | 88.5% | TODO | TODO |
| 32,768 | 1380.3 | 65.65 | 100.0% | TODO | TODO |
| 65,536 | TODO | 59.67 | 100.0% | TODO | TODO |
| 131,072 | 1287.5 | 61.19 | 100.0% | TODO | TODO |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp.svg)

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

## Multiple users

Aggregate delivered output tok/s. Context 4096, greedy, thinking off, 128
output tokens, `cache_prompt=false`, one cold cohort on a fresh server per
point. Workloads come from the
[corpus](../qwen3.8-27b/artifacts/speculative-corpus.json): `repetition`
runs `repetition_word`; `mixed` cycles distinct requests through
`expository_pangram`, `cpp_ring_buffer`, `reasoning_train`. `Exact` counts
llama.cpp completions whose hash matches the Gufo AR C1 reference.
Set `--sessions C --max-pending-per-client 8` when benchmarking up to eight
requests from one host.

Gufo values measured 2026-09-20 as the **sum of individual request decode
rates** (`completion_tokens / decode_seconds` from server-reported decode
time, excluding prefill and waiting between decode steps). The HTTP refresh
that uses the corpus aggregate metric on both sides is **TODO**; until then
the Gufo and llama.cpp columns are not the same metric.

<!-- bench:multi-repetition -->
| Users | Gufo AR | llama.cpp AR | Gain | Gufo MTP | Gain vs llama.cpp AR | Exact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 27.3 | TODO | TODO | 84.6 | TODO | TODO |
| 2 | 46.4 | TODO | TODO | 127.7 | TODO | TODO |
| 4 | 76.7 | TODO | TODO | 166.1 | TODO | TODO |
| 6 | 95.4 | TODO | TODO | 181.2 | TODO | TODO |
| 8 | 108.5 | TODO | TODO | **188.9** | TODO | TODO |
<!-- /bench -->

![Multiple users, repetition](artifacts/charts/multi-repetition.svg)

<!-- bench:multi-mixed -->
| Users | Gufo AR | llama.cpp AR | Gain | Gufo MTP | Gain vs llama.cpp AR | Exact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 27.0 | TODO | TODO | 49.3 | TODO | TODO |
| 2 | 43.2 | TODO | TODO | 66.4 | TODO | TODO |
| 4 | 66.8 | TODO | TODO | 95.2 | TODO | TODO |
| 6 | 78.2 | TODO | TODO | 110.4 | TODO | TODO |
| 8 | 86.7 | TODO | TODO | 118.3 | TODO | TODO |
<!-- /bench -->

![Multiple users, mixed corpus](artifacts/charts/multi-mixed.svg)

MTP acceptance is 100% on repetition and 74.2–84.9% on the mixed cohorts.
Every completion matches AR; all cohorts report zero cache hits.
Greedy timing-based depth choices can vary between runs.
MTP rollback storage grows on demand to about 147 MiB per session at seven
drafts; KV caches and other session state are additional.

**C1 is a single user.** The HTTP and raw CLI prompts above differ.
Compare identical prompts and timing boundaries.
`gufo bench` is C1; the published tables come from `tools/bench/model-bench.py`.

## Memory

GPU-visible unified allocation, C1, context capacity 133121, excluding
separate CPU memory: **TODO**. llama.cpp resident device memory at the same
`-c`: **TODO**.

<!-- bench:memory -->
| Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | TODO | TODO | TODO |
| 16K prefix, pp4096 + tg128 | TODO | TODO | TODO |
<!-- /bench -->

## Image encoder

Warm `mmproj-BF16.gguf` encoding; excludes preprocessing, first weight upload
and language-model prefill. Complete embeddings are byte-identical to the
previous encoder and reproduce exactly across runs. llama.cpp `--mmproj`
encode with the same file: **TODO**.

<!-- bench:image-encoder -->
| RGB image | Merged tokens | Gufo ms | llama.cpp ms | Gain |
| --- | ---: | ---: | ---: | ---: |
| 256×256 | 64 | 21.3 | TODO | TODO |
| 1024×1024 | 1024 | 1249 | TODO | TODO |
<!-- /bench -->

![Image encoder](artifacts/charts/image-encoder.svg)

The 4096-patch attention specialization saves 24 MiB of score/probability
scratch. Other image shapes retain their original tile layout.
