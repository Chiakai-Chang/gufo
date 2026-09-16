# Gufo: the Strix Halo inference engine

<p align="center">
  <img src="assets/gufo-logo.jpg" alt="Gufo logo" width="180">
</p>

Gufo is a vertical local inference engine specifically built and optimized for the AMD Strix Halo hardware:
Ryzen AI MAX+ 395 systems with Radeon 8060S (`gfx1151`), an XDNA2 NPU, and up to 128 GiB of unified memory.

Supported models:

- antirez's [DeepSeek-V4-Flash IQ2XXS](https://huggingface.co/antirez/deepseek-v4-gguf/blob/main/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf) GGUF with [DSpark](https://huggingface.co/antirez/deepseek-v4-gguf/blob/main/DeepSeek-V4-Flash-DSpark-support-0731.gguf) support.
  With its MoE architecture, 284B parameters (13B active), and quantization-aware training techniques, it is the largest and smartest text model that this hardware can run without losing too much of its full-quality accuracy.
- [Qwen3.8-27B:UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-UD-Q4_K_XL.gguf) and [Qwen3.8-27B-UD-Q8_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-UD-Q8_K_XL.gguf) GGUFs from unsloth with z-lab's [DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF). The best choice when you can't saturate your unified memory and want to leave room for something else.
- [Qwen3.8-Flash-Next:UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) GGUF from unsloth with its shared Q8 MTP draft block. This 103.7 GiB is a good trade-off between DeepSeek and Qwen when you still need some space for something while having a capable model running.
- [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) Safetensors text to video generation model.
  Even though the community has built quicker implementations, as of now we decided to support just MiniMaxAI's official one to retain the full model quality.
- [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) Safetensors for audio to text.
- [Qwen3-TTS](https://huggingface.co/collections/Qwen/qwen3-tts) 1.7B models: Base, CustomVoice, and VoiceDesign.

More info in [MODELS.md](./docs/MODELS.md)

## Benchmarks

### Text

#### Qwen3.8-27B Q4

**Date: 2026-09-16 — Gufo `7daf2ebc` — llama.cpp benchmark binary `169e4a7ff` (build 10489)**

Gufo and `llama.cpp` were benchmarked on the same Strix Halo machine and the same `UD-Q4_K_XL` model file, with FP16 KV caches. Both use 2,048 prompt tokens and 128 generation steps at each prepared context depth. Their benchmark tools use different token sequences: Gufo greedily generates output, while `llama-bench` feeds random tokens. The Gufo / llama.cpp figures compare throughput at the same workload shape, not identical completions. Gufo used three repetitions; llama.cpp used one. Values are tokens/s.

| Depth | Gufo `pp2048 / tg128` | llama.cpp `pp2048 / tg128` | Gufo / llama.cpp `pp2048 / tg128` |
| ----: | --------------------: | -------------------------: | --------------------------------: |
|     0 |        689.69 / 12.02 |             348.41 / 12.11 |                        198% / 99% |
|   16K |        523.59 / 11.32 |             252.01 / 11.46 |                        208% / 99% |
|   32K |        354.36 / 10.58 |             194.40 / 10.89 |                        182% / 97% |
|   64K |         199.97 / 9.25 |              121.42 / 9.94 |                        165% / 93% |
|  128K |         106.52 / 6.35 |               76.95 / 8.45 |                        138% / 75% |

On 2026-09-17, the Q4 target and `DFlash2-Q4_K_M` draft were run on the [corpus suite](benchmarks/qwen3.8-27b/speculative-corpus.json) with production chat framing, greedy decoding, a 128-token output limit, and one run per case. These short prompts have no prepared context depth, so their results are separate from the depth sweep. “Mixed” contains nine non-repetitive prompts; “repetition” contains the two repetitive prompts. Throughput is total generated tokens divided by total generation time within each group. Acceptance is accepted draft tokens divided by proposed draft tokens. All 11 DFlash2 completions matched Gufo's non-speculative output token for token.

| Corpus group | Cases | Gufo AR `tg≤128` | Gufo DFlash2 `tg≤128` | Acceptance | DFlash2 / Gufo AR | llama.cpp AR `tg≤128` | llama.cpp / Gufo exact |
| -----------: | ----: | ---------------: | --------------------: | ---------: | ----------------: | ---------------------: | ----------------------: |
|        Mixed |     9 |            11.69 |                 29.26 |      48.3% |             2.50x |                  12.00 |                     3/9 |
|   Repetition |     2 |            11.72 |                 53.83 |      89.6% |             4.59x |                  11.99 |                     1/2 |

The corpus llama.cpp run used the supplied nixpkgs `llama-completion` binary (version `7d56da7`, package `llama-cpp-10063`) and the same Q4 target, prompt tokens, and greedy output limit. Its reported rate is decode evaluation tokens divided by decode evaluation time, excluding the first-token evaluation. Four of its 11 completion texts exactly matched Gufo's. The other seven diverged in generated content despite identical prompt tokens, so no DFlash2 / llama.cpp speedup is claimed from this corpus.

With concurrency token generation using the dflash2 drafter (cumulative)

_Illustrative facsimile data; these are not measured benchmark results._

| Concurrency | Gufo `pp2048/tg128` | Over llama.cpp `pp2048/tg128` | DFlash2 mixed corpus/repetition `tg128` | DFlash2 over llama.cpp `tg128` |
| ----------: | ------------------: | ----------------------------: | --------------------------------------: | -----------------------------: |
|           1 |       545.15 / 7.10 |                   155% / 156% |                           36.46 / 62.06 |                   801% / 1364% |
|           2 |       529.80 / 6.90 |                   152% / 157% |                           41.85 / 84.07 |                   951% / 1911% |
|           4 |       503.40 / 6.70 |                   147% / 160% |                           46.48 / 94.43 |                  1107% / 2248% |
|           6 |       476.60 / 6.40 |                   142% / 160% |                           55.02 / 96.06 |                  1376% / 2402% |
|           8 |       451.20 / 6.20 |                   141% / 163% |                          56.73 / 100.37 |                  1493% / 2641% |

#### DeepSeek V4 Flash

Comparison of gufo's native DeepSeek V4 Flash path against `dwarfstar` on the same hardware, the same model weights, and the same numeric precision.

_Illustrative facsimile data; these are not measured benchmark results._

| Depth | Gufo `pp2048/tg128` | Over llama.cpp `pp2048/tg128` | DFlash2 mixed corpus/repetition `tg128` | DFlash2 over llama.cpp `tg128` |
| ----: | ------------------: | ----------------------------: | --------------------------------------: | -----------------------------: |
|     0 |       545.15 / 7.10 |                   155% / 156% |                              62.06 / 50 |                  1364% / 1099% |
|   16K |       446.94 / 6.50 |                   166% / 176% |                              62.06 / 50 |                  1677% / 1351% |
|   32K |       398.20 / 6.20 |                   173% / 188% |                              62.06 / 50 |                  1881% / 1515% |
|   64K |       351.60 / 5.90 |                   185% / 200% |                              62.06 / 50 |                  2104% / 1695% |
|  128K |       289.40 / 5.50 |                   207% / 224% |                              62.06 / 50 |                  2533% / 2041% |

With concurrency token generation using the dflash2 drafter (cumulative) at depth 0

| Concurrency | Gufo `pp2048/tg128` | Over llama.cpp `pp2048/tg128` | DFlash2 mixed corpus/repetition `tg128` | DFlash2 over llama.cpp `tg128` |
| ----------: | ------------------: | ----------------------------: | --------------------------------------: | -----------------------------: |
|           1 |       545.15 / 7.10 |                   155% / 156% |                           36.46 / 62.06 |                   801% / 1364% |
|           2 |       529.80 / 6.90 |                   152% / 157% |                           41.85 / 84.07 |                   951% / 1911% |
|           4 |       503.40 / 6.70 |                   147% / 160% |                           46.48 / 94.43 |                  1107% / 2248% |
|           6 |       476.60 / 6.40 |                   142% / 160% |                           55.02 / 96.06 |                  1376% / 2402% |
|           8 |       451.20 / 6.20 |                   141% / 163% |                          56.73 / 100.37 |                  1493% / 2641% |

### Audio

**Date: 2026-09-10**

Comparison of gufo's native Qwen3-TTS and Qwen3-ASR paths against [audio.cpp](https://github.com/0xShug0/audio.cpp) on the same hardware, the same model weights (safetensors), and the same numeric precision.

| task | gufo RTF  | audio.cpp RTF | gufo faster |
| ---- | --------- | ------------- | ----------- |
| tts  | **0.472** | **0.663**     | **1.40x**   |
| asr  | **0.075** | **0.121**     | **1.62x**   |

For more information please see [TTS_ASR.md](./docs/benchmarks/TTS_ASR.md).

## Manifest/Philosophy

- The project is vertical on the AMD Strix Halo 128 GiB; our goal is solely to optimize it. This enables optimizations that otherwise wouldn't be possible if we were focusing on other chips as well. Smaller models should fit the 32 and 64 GiB hardware, but no test was conducted on them.
- Quality over speed: we want to squeeze the most out of this chip without compromising on quality compared to other available tools (llama.cpp, audio.cpp, dwarfstar, etc.). To guarantee this we ensure several steps during the development, such as logits checks, internal eval, and a harness + model evaluation framework (coming soon). If at some point a breakthrough novelty brings a lot of speed at the cost of a little accuracy, the feature would be opt-in and the user will be responsible for enabling it, acknowledging the accuracy degradation.
- We only support a few models to allow us to run extremely long optimization sessions to improve kernels based on the Strix Halo architecture. Models are selected based on evidence collected by the community on "the best model for task X for Strix Halo".
- Code duplication over code re-utilization across models and quants. Despite being counterintuitive, it allows us to make models evolve independently without huge refactors when an optimization works only for a model and not for another.
- We would like this project to be the reference for the community using Strix Halo, and every PR is welcome.
- We don't to sacrificate multi-agent scenarios, concurrent requests are a first class citizen gufo.

## Quickstart

```sh
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --repo-type model \
  --local-dir models/Qwen3.8-27B-GGUF
nix develop -c hf download z-lab/Qwen3.8-27B-DFlash2-GGUF \
  Qwen3.8-27B-DFlash2-Q8_0.gguf \
  --repo-type model \
  --local-dir models/Qwen3.8-27B-DFlash2-GGUF
podman pull ghcr.io/gufo-org/toolboxes/gufo-runtime:latest
podman run --rm \
  --userns=keep-id:uid=1000,gid=1000 \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add keep-groups \
  --ulimit memlock=-1 \
  -p 8080:8080 \
  -v ./models:/models:ro \
  ghcr.io/gufo-org/toolboxes/gufo-runtime:latest \
  gufo serve --host 0.0.0.0 --port 8080 llm \
  --model /models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --speculative dflash2 \
  --dflash-model /models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q8_0.gguf
```

Then, from another terminal, ask it something through the OpenAI-compatible API:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-27B-UD-Q8_K_XL",
    "messages": [{"role": "user", "content": "Say something"}]
  }'
```

The server also exposes `/v1/completions`, `/v1/responses`, `/v1/models`, and
`/health`. Any OpenAI-compatible client can point at `http://localhost:8080`.

## Supported Platform

Linux x86-64 on AMD Strix Halo (`gfx1151` GPU and XDNA2 NPU) is the only
planned production platform. Windows, macOS, and CUDA are out of scope.

Build with Nix only; direct host builds are unsupported:

```sh
nix build                          # build default package (gfx1151 + XRT)
./result/bin/gufo diagnose        # run hardware probe & diagnostics
./result/bin/gufo serve           # run server
nix build .#checks.x86_64-linux.pr # canonical PR test command (all gates)
```

For non-Nix users, [toolboxes](https://github.com/gufo-org/toolboxes) for Docker and Podman are available:

```sh
podman pull ghcr.io/gufo-org/toolboxes/gufo-runtime:latest
```

### Why Nix?

Nix provides a reproducible environment by having all the dependencies in a single file, the [flake.nix](./flake.nix). Our primary goal is to provide an engine that performs better on the Strix Halo without sacrificing accuracy. Nix helps us develop this project by removing all those variables that may make experiments not reproducible across several machines (different driver versions, different environment variables, etc.).

## Reference Projects

The initial design is informed by the following open source projects:

- `llama.cpp` for compact model serving, GGUF, and CPU/GPU correctness paths.
- `vLLM` for continuous batching and paged request scheduling.
- `hipEngine` for torch-free HIP execution, and native speculative-cycle work.
- `ROCmFPX` for activation-aware quantization and quality evaluation.
- `DS4` for DeepSeek V4 Flash, MoE scheduling, and DSpark.
- `ypapadop-amd/ggml` `hsa-backend` for XDNA2 HSA dispatch and MLIR-AIE
  integration patterns.
- `audio.cpp` for audio models for tts and asr tasks.
