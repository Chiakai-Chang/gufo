---
name: benchmark-model
description: Measure every retained number in docs/models/<model>/BENCHMARKS.md for one Gufo model on Strix Halo gfx1151 over HTTP, run the same workload against the reference project's OpenAI-compatible server (llama.cpp, audio.cpp, ...), and report the gain per cell.
metadata:
  origin: gufo
---

# Benchmark a model

Fill or refresh `docs/models/<model>/BENCHMARKS.md` for one model. Which
tables exist and which external project is the comparison baseline depend on
the model category below. Read the model's `README.md`, `BENCHMARKS.md` and
`EVALUATION.md` first, then [docs/BENCHMARKS.md](../../../docs/BENCHMARKS.md)
for methodology. Follow the user's machine, time and Git instructions.

## Start

Ask the user before measuring anything:

1. Which model (`docs/models/<model>`) to benchmark.
2. Where the model files are: target GGUF or safetensors directory per
   variant, draft/MTP/DSpark support files, mmproj, audio fixtures. Model
   files live wherever the user keeps them; nothing in Git records paths.
3. Full benchmark (every table, every cell, both targets) or only the cells
   currently `TODO` in `BENCHMARKS.md`.

Then confirm the time budget: a full LLM sweep with the reference is hours.

## Measurement model

Everything is measured **over HTTP on both sides** so the prompt, timed scope
and transport are identical: Gufo through `gufo serve`, the reference through
its own OpenAI-compatible server. `gufo bench` and the standalone CLIs are
kernel-iteration tools; their numbers are not published in the comparison
tables.

The driver is `tools/bench/model-bench.py`. `docs/models/<model>/artifacts/bench.json`
declares the category, variant ids, sweep grid, workloads and reference server
flags; model file paths are passed on the command line. The driver launches
the right server itself on a private port, runs one target at a time, writes
one JSON per table (and per mode for concurrency tables) into
`docs/models/<model>/artifacts/`, and renders the Markdown tables between
`<!-- bench:<table> -->` markers in `BENCHMARKS.md`:

```sh
nix build   # production Gufo binary at result/bin/gufo
FILES="--gguf q4=/path/to/target-q4.gguf --draft q4=/path/to/draft-q4.gguf --mmproj /path/to/mmproj.gguf"
nix develop -c python3 tools/bench/model-bench.py --model <model> tables            # ids and existing artifacts
nix develop -c python3 tools/bench/model-bench.py --model <model> $FILES run --target gufo
nix develop -c python3 tools/bench/model-bench.py --model <model> $FILES run --target reference
nix develop -c python3 tools/bench/model-bench.py --model <model> render           # or render --check
```

`run` flags: `--todo` measures only rows whose target-owned cells are `TODO`
in `BENCHMARKS.md`; `--table <id>[,<id>]` restricts to named tables;
`--drop-caches "<privileged command>"` is needed by the loading table on hosts
without passwordless sudo/doas (`sync; echo 3 > /proc/sys/vm/drop_caches`);
`--config` and `--artifacts-dir` point at an alternative `bench.json` and
output directory for experiments that must not touch the retained artifacts.
Server logs go to the ignored `artifacts/model-bench/`. `render` keeps
hand-entered cells that no artifact covers, so tables can hold both, and it
draws one SVG chart per table with data into
`docs/models/<model>/artifacts/charts/` (matplotlib, deterministic output),
placed as an image line right after the table. Tables stay the source of
truth for agents; charts are an addition for readers. `--no-charts` skips them.

`bench.json` (`schema: gufo-model-bench/1`) holds: `category`; `model.id`;
`files` (the file roles the CLI must supply, with a description each);
`variants` (id, label, required roles; single-variant models use `default`
and unsuffixed table ids); `sampling`; `speculative` (mode, label, Gufo
arguments with `{role}` placeholders, and which reference mode it is compared
against); `tables` keyed by table id with that table's workload, and
`gufo.serve` / `reference.server` + `args` for the two servers. Readiness is
`GET /ready` on Gufo and `GET /health` on llama-server.

Reference servers (llama.cpp, audio.cpp, stable-diffusion.cpp) come from this
repository's `flake.nix`; do not use ad-hoc installs.

Request-level timing comes from the response, not the client clock, whenever
the server provides it: Gufo and llama-server both return llama.cpp-compatible
`timings` (`prompt_n` = newly processed tokens, `cache_n` = reused tokens,
`prompt_ms`, `predicted_n`, `predicted_ms`); Gufo adds `usage.gufo` stage
metrics. Servers without timings get client-measured whole-request wall time,
and the table says so.

## Rules for every category

1. Production binary: `nix build`, `./result/bin/gufo`. Record the binary
   SHA-256 prefix, Git revision, model artifact identities (repo, revision,
   quantization, hashes), reference tool version/commit, backend (ROCm/Vulkan),
   full server command lines and the measurement date. Server flags go into
   the artifact `notes`; hosts, prompts, generated text and paths never do.
2. Same input, same artifact, same timed scope on both sides: same GGUF or
   safetensors, same prompt or audio file, same output length, same
   seed/temperature, same context depth, same concurrency, fresh server per
   point, no prompt-cache hits unless the cell says so. If the reference cannot
   match (different quantization, no equivalent speculative mode, no batching),
   measure the closest configuration and state the mismatch under the table.
3. Warm once, then time. One warmed sample per point is acceptable for a
   sweep; add repetitions when two runs differ by more than about 2% or a
   regression is suspected, and report mean ± sd. Run model jobs sequentially;
   nothing else on the GPU.
4. Run the model's quality checks in `EVALUATION.md` before publishing speed.
   Greedy speculative output must match AR token IDs; the reference's greedy
   output is compared by completion hash and the match count is reported. A
   mismatch or device fault is not a result.
5. A cell without a current qualified measurement is `TODO`, never a stale
   number. Keep dates per table; do not sum stage medians into a headline.
6. Retained JSON goes to `docs/models/<model>/artifacts/`; raw samples,
   traces, audio, images and videos stay in the ignored top-level `artifacts/`.

## Comparison columns

Every headline table carries the reference project next to Gufo:

```
| ... | Gufo <metric> | <ref> <metric> | Gain | ... |
```

- `Gain` is a percentage, positive when Gufo is better:
  - throughput (tok/s, audio s per wall s): `(Gufo / ref − 1) × 100`
  - latency, RTF, wall time, memory: `(ref / Gufo − 1) × 100`
- No difference column: both raw values are visible, and the extra column
  makes wide tables unreadable. When a cell is `mean ± sd`, Gain uses means.
- Name the reference in the header (`llama.cpp pp`, `audio.cpp RTF`).
- `TODO` on either side leaves `Gain` as `TODO`.
- When Gufo runs a mode the reference lacks (DFlash2, DSpark, MTP without a
  matching draft), compare against the reference's AR number and label the
  column `Gain vs <ref> AR`.
- One table per quantization and per mode; never pack `pp / tg` pairs or two
  quantizations into one cell.

Rendered tables sit between `<!-- bench:<table-id> -->` and
`<!-- /bench -->` markers so `render` can replace them in place; the prose
around them (dates, acceptance notes, caveats) is hand-maintained.

## LLM (GGUF text and vision models)

Models: `deepseek-v4-flash`, `qwen3.8-27b`, `qwen3.8-flash-next`.
Reference: **llama.cpp** `llama-server`, same GGUF, ROCm build,
`-ngl 999 -fa on --cache-reuse 0 --jinja --reasoning off`, `-np` equal to the
largest concurrency, `-c` covering the deepest sweep point plus 2048 + 128.
Gufo: `gufo serve --sessions 8 --max-pending-per-client 8`, thinking off,
greedy, seed 1. The sweep parameters are identical across the three models so
the documents stay comparable.

Tables, in order (table ids in parentheses; append `-<quant>` when the
model has several quantizations, e.g. `single-ar-q4`):

1. **Loading** (`loading`). Cold-file-cache launch to `/health` readiness
   (drop caches, start the server, poll), then Gufo snapshot size and restore
   time for the documented prompt. Reference: `llama-server` readiness with
   the same GGUF. `Target | Gufo ready | llama.cpp ready | Gain`.
2. **Single user, autoregressive** (`single-ar`). pp2048 / tg128, C1, depth
   `0,4096,8192,12288,16384,32768`; add `65536,131072` when the model's
   context allows. Depth `d` is a cached prefix: send the `d`-token prefix once
   with `cache_prompt=true`, then prefix + 2048 new tokens with 128 output
   tokens. Neither server exposes a tokenizer endpoint, so the driver
   calibrates the synthetic prefix text (`prefix.generator`, seeded) against
   each server's reported `prompt_n`/`cache_n` and accepts a point only when
   `cache_n` is within `depth_tolerance` of `d` and `prompt_n` within it of
   2048; the actual counts are stored in the artifact. The same text is sent
   to both servers.
   `Depth | Gufo pp | llama.cpp pp | Gain | Gufo tg | llama.cpp tg | Gain`.
3. **Single user, speculative** (`single-<spec>`, DFlash2 / DSpark / MTP as
   the model supports). Same grid plus `Acceptance` from `usage.gufo`.
   Reference: `llama-server --model-draft` when an equivalent draft exists
   (`llama.cpp draft tg | Gain`), otherwise llama.cpp AR
   (`llama.cpp AR tg | Gain vs llama.cpp AR`); the note under the table says
   which. `Depth | Gufo pp | Gufo tg | Acceptance | llama.cpp ... tg | Gain ...`.
4. **Multiple users** (`multi-mixed`, `multi-repetition`). `C = 1,2,4,6,8`,
   context capacity 4096, 128 output tokens, fresh server per point,
   `cache_prompt=false`. Two workloads from
   `docs/models/qwen3.8-27b/artifacts/speculative-corpus.json`: `repetition`
   and `mixed` (distinct layout). Metric: aggregate delivered output tok/s
   (`aggregate.output_tokens_per_second.overall`). `Exact` is the count of
   llama.cpp completions whose hash matches the Gufo AR C1 reference. Report
   C8 median / p95 latency and acceptance under the table.
   `Users | Gufo AR | llama.cpp AR | Gain | Gufo <spec> | Gain vs llama.cpp AR | Exact`.
5. **Memory** (`memory`). GPU-visible allocation at the documented context
   capacity for pp2048+tg128 and a 16K-prefix pp4096+tg128 (`usage.gufo`
   memory fields / `gufo diagnose`). Reference: `llama-server` resident
   device memory at the same `-c`, read from the ROCm SMI while the request
   runs. `Workload | Gufo GiB | llama.cpp GiB | Gain`.
6. **Image encoder** (`image-encoder`, vision models only). Warm encode
   latency for 256×256 and 1024×1024 RGB through the chat endpoint with the
   BF16 projector, isolated from language prefill via `usage.gufo` stages.
   Reference: `llama-server --mmproj` with the same file; whole-request wall
   when it reports no encode stage.
   `RGB image | Merged tokens | Gufo ms | llama.cpp ms | Gain`.

Concurrency artifacts are `gufo-serving-bench` corpus reports named
`<table>-gufo-ar.json`, `<table>-gufo-<spec>.json` and `<table>-reference.json`;
the other tables use the compact `model-bench-table` schema with one entry per
row and the actual `cache_n`/`prompt_n` counts. The image-encoder table is not
automated yet; leave its cells `TODO` or fill them by hand from a documented
measurement.

## ASR (audio to text)

Model: `qwen3-asr`. Reference: **audio.cpp** server (`qwen3_asr` model spec,
ROCm), `/v1/audio/transcriptions`, same WAV, greedy. Both transcripts are
compared against the official reference transcript.

1. **Loading.** Cold-file-cache readiness; first-request and warm latency for
   a short clip.
2. **Single request.** The documented fixed recording (15.05 s English), one
   warmup, three timed requests, median. Rows: request time, RTF (request /
   input duration, lower is better), audio seconds per wall second; Gufo-only
   rows for audio-encoder time and text prefill+generation from `usage.gufo`.
   Columns: `Metric | Gufo | audio.cpp | Gain`.
3. **Concurrent HTTP.** C = 1,2,4,6,8, same recording: per-request latency
   median / p95 and aggregate audio seconds per wall second, both servers.
4. **Long-form.** One recording over ten minutes: wall time, RTF, transcript
   diff between the two implementations and against the reference.
5. **Profile** (Gufo only). Dispatches, GPU work, GPU-busy share, projection
   share from `tools/prof/prof.py --stages qwen-asr`, run separately.

## TTS (text and reference audio to speech)

Model: `qwen3-tts`. Reference: **audio.cpp** server (`qwen3_tts` model spec,
ROCm), `/v1/audio/speech`, same text, seed, sampling and variant
(CustomVoice / VoiceDesign / Base ICL); variants audio.cpp does not expose
are `n/a`.

1. **Loading.** Cold-file-cache readiness per variant; first-request and warm
   latency for a short sentence.
2. **Single request.** The documented 37-word paragraph, 64 codec frames
   (5.12 s), one warmup, three timed, median. Per variant:
   `Variant | Gufo WAV | audio.cpp WAV | Gain | Gufo RTF | audio.cpp RTF | Gain | First PCM | Complete PCM`.
   RTF = request time / generated audio duration. Gufo streamed and buffered
   PCM must be identical; seeded runs must repeat exactly. First-audio latency
   is measured on the streaming endpoint of each server when it has one.
3. **Transports** (Gufo only). PCM HTTP, SSE, WebSocket first-audio and
   complete times; byte-identical PCM across transports.
4. **Concurrency.** C = 1,2,4,6,8 completion times on both servers;
   execution is serialized, report the queue effect honestly.
5. **Profile** (Gufo only). Dispatches, GPU work, GPU busy, projection share
   per variant from `tools/prof/prof.py --stages qwen-tts`.

Natural-EOS duration and intelligibility belong to `EVALUATION.md`.

## Image generation and editing

Model: `qwen-image-2.1`. Reference: **stable-diffusion.cpp** server on ROCm
(`/v1/images/generations`) with the same checkpoint when it supports the
model; otherwise an OpenAI-compatible wrapper around the official Diffusers
pipeline on ROCm, labelled `Diffusers (ROCm)`. Same prompt, seed, size,
steps, scheduler and guidance.

1. **Loading.** Metadata readiness, first generation including component
   loading, and the same request warm.
2. **Generation and edit, C1.** 1024×1024, the default 40 steps, seed 42.
   Rows: generation; edit with one reference image (edit only where the
   reference supports it). Columns:
   `Mode | Steps | Prompt (s) | Denoising (s) | VAE (s) | Gufo total | <ref> total | Gain`;
   stage columns are Gufo `usage.gufo` values. Add 512×512 / 20 steps as a
   second control when the reference is slow.
3. **GPU split** (Gufo only). Native projections, fused feed-forward, fused
   attention, convolution and idle share from a separate request-only profile.
4. **Concurrency.** Requests are queued; report C2 completion times and state
   that no batching is implemented.

A two-step smoke check is a development control, never the headline. Image
quality is measured in `EVALUATION.md`.

## Video / audiovisual generation

Model: `minimax-h3`. Reference: **audio.cpp** server (`minimax_h3` model
spec) if it executes on ROCm; verify first and fall back to a wrapper around
the official Diffusers pipeline, labelled. Full generation requires the user's
explicit approval and `--allow-full-generation`.

1. **Loading.** Metadata readiness; first prompt-encoder layers time; peak
   retained device memory.
2. **Presets.** Per preset (`exact 512x512`, `exact-1344x768`, `fast`,
   `aggressive`): single-request wall, `first_preview_ms`, peak resident
   memory, delivery-quality note. Columns:
   `Preset | Gufo wall | <ref> wall | Gain | Gufo peak GiB | <ref> peak GiB | Gain`.
3. **Phase split** (Gufo only). Prompt encoding, AdaLN precompute, denoising,
   VisualVAE, AudioVAE, mux, from `tools/gufo/h3_profile.py`.
4. **Component inventory.** Tensor bytes per component; never summed as if
   resident together.
5. **Queue.** One worker; report C2 queue latency and say there is no
   parallel execution.

## Finish

Render the tables, check that every Gufo/reference pair used the same
workload identity, update dates and the reproduction block, and list the
remaining `TODO` cells with the reason (tool missing, reference unsupported,
time budget). Update `EXPERIMENTS.md` only when a measurement changes a
retained decision. Summarize per table: Gufo, reference, best and worst gain,
and every cell where completion hashes or transcripts did not match.
