# Qwen3.8-Flash-Next on Strix Halo

| | |
| --- | --- |
| Host | Linux x86-64, AMD `gfx1151`, 128 GB unified memory |
| Gufo | `nix build` at revision `89d58eb7`, binary SHA-256 `4b1e592acb7e4a56…` |
| Target | `unsloth/Qwen3.8-Flash-Next-GGUF` revision `38bb39ee`, **UD-Q4_K_XL** (four shards, 103.7 GiB) |
| Speculative | MTP sidecar `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, same revision, adaptive up to 7 proposals |
| Reference | llama.cpp `llama-server` release `b11069` (`0.4.1-dev (build 11069, commit 68d9053a)`), ROCm gfx1151 from `flake.nix`, same GGUF. MTP cells use `llama-server-mtp`, the same build recipe on the open [ggml-org/llama.cpp#28243](https://github.com/ggml-org/llama.cpp/pull/28243) branch (commit `6fcaa16f`), because the release cannot load the shared MTP sidecar |
| Method | HTTP on both servers, same prompts and timed scope, greedy, thinking off, one sample per point, fresh server per table and concurrency level; `tools/bench/model-bench.py`, 2026-09-22. Every artifact records its server command |
| Gain | Gufo over llama.cpp, positive when Gufo is better |
| Identities | [`artifacts/model-identities.json`](artifacts/model-identities.json) |
| Layout | [benchmark-model skill](../../../.agents/skills/benchmark-model/SKILL.md) |

Chat follows the official template (thinking on, `xhigh` effort, prior
reasoning preserved); the benchmarks run with thinking off. PNG/JPEG input
works with AR/MTP and `mmproj-BF16.gguf`; see
[image usage and validation](README.md#images). **Original
unquantized-model and GGUF-conversion parity remain unqualified.** This
model has no fast correctness suite; the driver's checks (128 generated
tokens per single-user sample, MTP completion hashes against the Gufo AR C1
reference) are the correctness gates of this refresh. Unmeasured cells are
**TODO**; cells this host cannot produce for the reference (a hardware
limit, not a missing feature) are **n/a**, with the reason next to the table.

## Loading

Cold-file-cache launch to `/ready`. Gufo loaded
the four shards plus the MTP sidecar with two sessions at context capacity
262144. llama.cpp loaded the same shards at `-c 35456` (the capacity of its
depth rows): at `-c 262144` it preallocates the whole KV cache for this
104 GiB model and the kernel OOM-killed it before readiness, so its number
is for a smaller window. Gufo maps the weights lazily, which is why its
cold and warm readiness differ little.

<!-- bench:loading -->
| Target | Gufo ready | llama.cpp ready | Gain |
| --- | ---: | ---: | ---: |
| UD-Q4_K_XL | 16.30 s | 156.35 s | +859.2% |
<!-- /bench -->

![Cold-file-cache launch to readiness](artifacts/charts/loading.svg)

Warm-cache readiness for comparison: Gufo logged `load_completed` after
**15.6 s** (AR, context 133760) and **16.2 s** (MTP), with
`gpu_device_used_mib` 86940 (AR) and 89892 (MTP). A 4084-token AR prompt
snapshot occupied 212 MiB; restore/replay qualification is in
[evaluation](EVALUATION.md).

## Single user, autoregressive

Standard sweep: **pp2048 / tg128**, C1, greedy. Cells are tok/s from the
servers' own `timings` (`prompt_n` / `prompt_ms`, `predicted_n` /
`predicted_ms`). Depth is a cached conversation prefix: the driver sends a
user turn of about `d` tokens answered with an 8-token generated reply, then
the timed request continues that conversation with a new ~2048-token user
turn that asks for a long continuation, and 128 output tokens. Gufo ran
every depth at context capacity 133760. llama.cpp ran d0–32K at 35456 and
64K at 68224 because at `-c 133760` its resident set (103.7 GiB of mapped
weights plus 33 GiB of KV and compute buffers) exceeded the 125 GiB of RAM
and the kernel OOM-killed it during the 12K prefix; the 128K row is
therefore **n/a for llama.cpp on this host**.
Artifacts: `artifacts/single-ar-gufo.json`, `artifacts/single-ar-reference.json`.

<!-- bench:single-ar -->
| Depth | Gufo pp | llama.cpp pp | Gain | Gufo tg | llama.cpp tg | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1628.52 | 490.33 | +232.1% | 26.04 | 22.11 | +17.8% |
| 4,096 | 1523.39 | 428.50 | +255.5% | 25.98 | 21.18 | +22.7% |
| 8,192 | 1499.13 | 422.44 | +254.9% | 25.91 | 20.35 | +27.3% |
| 12,288 | 1477.98 | 395.59 | +273.6% | 25.46 | 18.72 | +36.0% |
| 16,384 | 1457.16 | 378.65 | +284.8% | 25.20 | 18.96 | +32.9% |
| 32,768 | 1421.93 | 306.77 | +363.5% | 24.33 | 16.07 | +51.4% |
| 65,536 | 1304.01 | 224.90 | +479.8% | 23.21 | 11.86 | +95.7% |
| 131,072 | 1292.02 | n/a | n/a | 21.93 | n/a | n/a |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Single user, MTP

Same workload with `--speculative mtp --mtp-model <mtp>` (adaptive, up to 7
proposals). Gufo pp includes predictor catch-up. `accepted/step` is the mean
number of accepted draft tokens per verification step
(`draft_tokens_accepted / (completion_tokens − draft_tokens_accepted)`);
tokens per step is this plus one. The measured turn asks for a detailed
summary and a story, so the generated text is ordinary prose.
Artifact: `artifacts/single-mtp-gufo.json`.

<!-- bench:single-mtp -->
| Depth | Gufo pp | llama.cpp pp | Gain | Gufo tg | llama.cpp tg | Gain | Gufo accepted/step | llama.cpp accepted/step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1438.69 | 444.77 | +223.5% | 32.18 | 31.58 | +1.9% | 0.64 | 1.46 |
| 4,096 | 1304.23 | 418.21 | +211.9% | 33.45 | 30.12 | +11.1% | 0.97 | 1.46 |
| 8,192 | 1324.72 | n/a | n/a | 33.32 | n/a | n/a | 0.91 | n/a |
| 12,288 | 1317.14 | n/a | n/a | 34.06 | n/a | n/a | 1.06 | n/a |
| 16,384 | 1298.17 | n/a | n/a | 33.09 | n/a | n/a | 0.94 | n/a |
| 32,768 | 1278.93 | n/a | n/a | 28.70 | n/a | n/a | 0.71 | n/a |
| 65,536 | 1176.17 | n/a | n/a | 28.69 | n/a | n/a | 1.13 | n/a |
| 131,072 | 1191.27 | n/a | n/a | 26.31 | n/a | n/a | 1.06 | n/a |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp.svg)

Repetitive workload: same prefixes and depths, but the measured turn asks the
model to repeat the passage word for word, so the output is fully predictable
— the single-user analogue of the `repetition` corpus below. Gufo's adaptive
controller is tuned to push hard on this case.

<!-- bench:single-mtp-repetition -->
| Depth | Gufo pp | llama.cpp pp | Gain | Gufo tg | llama.cpp tg | Gain | Gufo accepted/step | llama.cpp accepted/step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1605.30 | 459.03 | +249.7% | 59.39 | 44.00 | +35.0% | 3.74 | 2.66 |
| 4,096 | 1509.49 | 415.97 | +262.9% | 48.07 | 45.88 | +4.8% | 2.28 | 2.66 |
| 8,192 | 1472.04 | 388.24 | +279.2% | 50.19 | 37.01 | +35.6% | 2.46 | 2.76 |
| 12,288 | 1451.29 | n/a | n/a | 40.07 | n/a | n/a | 1.17 | n/a |
| 16,384 | 1429.19 | n/a | n/a | 50.57 | n/a | n/a | 2.76 | n/a |
| 32,768 | 1389.83 | n/a | n/a | 45.25 | n/a | n/a | 2.56 | n/a |
| 65,536 | 1202.92 | n/a | n/a | 37.23 | n/a | n/a | 2.28 | n/a |
| 131,072 | 1272.39 | n/a | n/a | 39.20 | n/a | n/a | 2.46 | n/a |
<!-- /bench -->

![Single user, MTP, repetitive](artifacts/charts/single-mtp-repetition.svg)

The llama.cpp MTP columns come from **`llama-server-mtp`**, built in
`flake.nix` from the open pull request
[ggml-org/llama.cpp#28243](https://github.com/ggml-org/llama.cpp/pull/28243)
(commit `6fcaa16f`, same ROCm build recipe as the release): release `b11069`
rejects the shared sidecar (`check_tensor_dims: tensor 'token_embd.weight'
not found`). Replace it with the release pin once the change is merged.
With the MTP draft loaded, llama.cpp's resident set exceeds the host's RAM
from the 12K prefix on (the kernel OOM-kills it at `-c 35456`), so the
deeper rows are **n/a** for llama.cpp on this host; a d8192 prose sample
that completed under swap pressure (1.66 tok/s) is excluded.

On generic prose Gufo MTP accepts 0.6–1.1 draft tokens per step and decodes
**32.2 tok/s** at d0 (+24% over Gufo AR) and **26.3 tok/s** at 128K (+20%);
llama.cpp MTP accepts 1.46 per step and decodes 31.6 / 30.1 tok/s at d0 /
4K, so the two are within 2–11% where both run. MTP costs Gufo 8–12% of
prefill throughput. On the repetitive workload Gufo MTP accepts 2.3–3.7 per
step and decodes **59.4 tok/s** at d0 and **39.2 tok/s** at 128K (2.3× /
1.8× Gufo AR); llama.cpp MTP reaches 44–46 tok/s at d0–4K with 2.7 per
step, so Gufo leads by 5–36% on the shared depths.

`gufo bench` controls (CLI, raw prefix, not HTTP; 2026-09-21): sampled MTP at
d0, seed 1, tg128, top-k/top-p/min-p disabled:

| Sampling | MTP tok/s | Acceptance |
| --- | ---: | ---: |
| T=0.7 | 39.83 | 65.2% |
| T=1.0 | 28.45 | 56.5% |

## Multiple users

Aggregate delivered output tok/s (`aggregate.output_tokens_per_second.overall`:
total output tokens divided by the sum of measured request-group spans).
Context 4096, greedy, thinking off, 128 output tokens,
`cache_prompt=false`, one warm-up round, one cold cohort on a fresh server
per point (`gufo serve --sessions C`; `llama-server -np C -c 4096·C`).
Workloads come from the
[corpus](../qwen3.8-27b/artifacts/speculative-corpus.json): `repetition`
runs `repetition_word`; `mixed` cycles distinct requests through
`expository_pangram`, `cpp_ring_buffer`, `reasoning_train`. `Exact` counts
llama.cpp AR completions whose hash matches the Gufo AR C1 reference.
Artifacts: `artifacts/multi-<table>-gufo-ar.json`, `-gufo-mtp.json`,
`-reference.json`.

<!-- bench:multi-repetition -->
| Users | Gufo AR | llama.cpp AR | Gain | Gufo MTP | llama.cpp MTP | Gain | Exact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 26.10 | 21.42 | +21.8% | 76.86 | 44.98 | +70.9% | 1/1 |
| 2 | 43.38 | 38.34 | +13.1% | 113.15 | 69.55 | +62.7% | 2/2 |
| 4 | 68.94 | 60.78 | +13.4% | 138.82 | 79.81 | +73.9% | 4/4 |
| 6 | 83.87 | 72.91 | +15.0% | 145.07 | 91.03 | +59.4% | 6/6 |
| 8 | 94.19 | 78.45 | +20.1% | 145.43 | n/a | n/a | 8/8 |
<!-- /bench -->

![Multiple users, repetition](artifacts/charts/multi-repetition.svg)

<!-- bench:multi-mixed -->
| Users | Gufo AR | llama.cpp AR | Gain | Gufo MTP | llama.cpp MTP | Gain | Exact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 25.87 | 21.56 | +20.0% | 44.66 | 38.94 | +14.7% | 1/3 |
| 2 | 40.43 | 33.88 | +19.3% | 62.66 | 50.89 | +23.1% | 2/4 |
| 4 | 58.41 | 51.80 | +12.8% | 84.97 | 53.16 | +59.8% | 2/4 |
| 6 | 71.69 | 61.49 | +16.6% | 91.29 | n/a | n/a | 4/6 |
| 8 | 78.30 | 67.49 | +16.0% | 95.43 | n/a | n/a | 3/8 |
<!-- /bench -->

![Multiple users, mixed corpus](artifacts/charts/multi-mixed.svg)

The llama.cpp MTP column uses `llama-server-mtp` (see the single-user MTP
table). It is OOM-killed at C8 on `repetition` and from C6 on `mixed`
(`-c 4096·C` plus the draft exceed the host's RAM), so those cells are
**n/a**. Where both run, Gufo MTP delivers +59…+74% on `repetition` and
+15…+60% on `mixed` (Gufo 1.4–4.3 accepted per step vs llama.cpp 2.4–2.9;
Gufo AR alone is +13…+22% over llama.cpp AR).

Gufo honours `cache_prompt=false`: all twenty Gufo cohorts and all ten
llama.cpp cohorts report **zero prompt-cache hits** (`render` prints
`cache hits 0/N` for every artifact). Every
Gufo AR and MTP completion at C2–C8 matches the Gufo AR C1 hashes (100%
exact on both corpora). llama.cpp matches Gufo on every `repetition`
completion but on only one of the three `mixed` prompts (two at C6; `Exact` 1/3 …
4/6): its greedy continuations of the other prompts diverge from Gufo's, which
is a numerical difference between implementations, not a benchmark defect;
timings are still comparable because every request produced 128 tokens.
MTP accepts 4.3 draft tokens per step on `repetition` and 1.4 on `mixed` at C8.
C8 request latency (median / p95): repetition Gufo AR 10.70 / 10.86 s, Gufo
MTP 6.68 / 7.01 s, llama.cpp AR 13.05 / 13.05 s; mixed Gufo AR 12.89 /
13.06 s, Gufo MTP 9.89 / 10.65 s, llama.cpp AR 15.17 / 15.17 s.

**C1 is a single user.** Its prompts differ from the single-user table's
synthetic 2048-token turn, so the two C1 rates are not the same measurement.

## Memory

Peak device-global HIP memory in use (`hipMemGetInfo` total − free, the
counter Gufo logs as `gpu_device_used_mib`), sampled every 250 ms by the
driver while the request runs; both servers autoregressive at context
capacity 133121, no projector loaded. The idle value before either server
started was 2.38 GiB and is included in both cells. Artifacts:
`artifacts/memory-gufo.json`, `artifacts/memory-reference.json`.

<!-- bench:memory -->
| Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 85.55 | 84.84 | -0.8% |
| 16K prefix, pp4096 + tg128 | 86.27 | 85.31 | -1.1% |
<!-- /bench -->

![GPU-visible allocation](artifacts/charts/memory.svg)

llama-server preallocates its KV cache at `-c`, so its footprint barely
grows with the prefix; Gufo's grows with the retained prompt state. The two
servers are within 1% of each other at this capacity.

## Image encoder

Prefill time (`prompt_ms`) of a chat request carrying one gradient PNG and
a one-line text turn, `max_tokens` 1, `cache_prompt=false`, one warm-up then
three timed samples per size (mean ± sd), both servers loading
`mmproj-BF16.gguf` (Gufo `--mmproj`, llama.cpp `--mmproj`). The time covers
the projector encode plus the prefill of the 64 / 1024 image tokens and the
text tokens (85 / 1045 prompt tokens on both servers); the encode alone is
not separable over HTTP, so the earlier hand-measured encode-only figures
(21.3 ms / 1249 ms, 2026-09-20) are superseded and not comparable.

<!-- bench:image-encoder -->
| RGB image | Merged tokens | Gufo ms | llama.cpp ms | Gain |
| --- | ---: | ---: | ---: | ---: |
| 256×256 | 64 | 253.9 ± 26.7 | 636.5 ± 106.6 | +150.7% |
| 1024×1024 | 1024 | 2078.6 ± 104.3 | 3212.2 ± 5.5 | +54.5% |
<!-- /bench -->

![Image encoder](artifacts/charts/image-encoder.svg)

Complete embeddings are byte-identical to the previous encoder and reproduce
exactly across runs. The 4096-patch attention specialization saves 24 MiB of
score/probability scratch. Other image shapes retain their original tile
layout.

## Reproduction

```sh
nix build
FILES="--gguf /path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
       --mtp /path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
nix develop -c python3 tools/bench/model-bench.py --model qwen3.8-flash-next tables
nix develop -c python3 tools/bench/model-bench.py --model qwen3.8-flash-next $FILES \
  run --target gufo --fresh --table single-ar,single-mtp,multi-repetition,multi-mixed,memory
nix develop -c python3 tools/bench/model-bench.py --model qwen3.8-flash-next $FILES \
  run --target reference --fresh --table single-ar,multi-repetition,multi-mixed,memory
nix develop -c python3 tools/bench/model-bench.py --model qwen3.8-flash-next render
# Loading needs --drop-caches "<privileged command>"; single-mtp on the
# reference fails to load the sidecar with llama.cpp b11069.
```

Until the driver's prefix turn is fixed, the single-user and memory runs
stop at the first depth row; run them with `--todo` after blanking only the
d0 / `pp2048 + tg128` rows, or expect the abort after d0 has been stored.
