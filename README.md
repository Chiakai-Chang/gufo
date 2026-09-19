# Gufo: the Strix Halo inference engine

<p align="center">
  <img src="assets/gufo-logo.jpg" alt="Gufo logo" width="180">
</p>

Gufo is a vertical local inference engine specifically built and optimized for the AMD Strix Halo hardware:
Ryzen AI MAX+ 395 systems with Radeon 8060S (`gfx1151`), up to 128 GiB of unified memory.

## Models and benchmarks

All model documentation lives under [docs/models](docs/models/README.md):

| Model | Inference modes | Performance and quality |
| --- | --- | --- |
| [Qwen3.8 27B](docs/models/qwen3.8-27b/README.md) | Q4/Q8, images, AR, DFlash2, native MTP | [Benchmarks](docs/models/qwen3.8-27b/BENCHMARKS.md) · [Evaluation](docs/models/qwen3.8-27b/EVALUATION.md) |
| [Qwen3.8 Flash-Next](docs/models/qwen3.8-flash-next/README.md) | Images, AR, MTP | [Benchmarks](docs/models/qwen3.8-flash-next/BENCHMARKS.md) · [Evaluation](docs/models/qwen3.8-flash-next/EVALUATION.md) |
| [DeepSeek V4 Flash](docs/models/deepseek-v4-flash/README.md) | AR, DSpark | [Benchmarks](docs/models/deepseek-v4-flash/BENCHMARKS.md) · [Evaluation](docs/models/deepseek-v4-flash/EVALUATION.md) |
| [Qwen3-ASR](docs/models/qwen3-asr/README.md) | Speech recognition | [Benchmarks](docs/models/qwen3-asr/BENCHMARKS.md) · [Evaluation](docs/models/qwen3-asr/EVALUATION.md) |
| [Qwen3-TTS](docs/models/qwen3-tts/README.md) | CustomVoice, VoiceDesign, Base cloning | [Benchmarks](docs/models/qwen3-tts/BENCHMARKS.md) · [Evaluation](docs/models/qwen3-tts/EVALUATION.md) |
| [MiniMax H3](docs/models/minimax-h3/README.md) | Text to video/audio, exact and approximate presets | [Benchmarks](docs/models/minimax-h3/BENCHMARKS.md) · [Evaluation](docs/models/minimax-h3/EVALUATION.md) |

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

Linux x86-64 on AMD Strix Halo (`gfx1151` GPU) is the only
planned production platform. Windows, macOS, and CUDA are out of scope.

Build with Nix only; direct host builds are unsupported:

```sh
nix build                          # build default package (gfx1151)
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
- `DS4` for DeepSeek V4 Flash, MoE scheduling, and DSpark.
- `audio.cpp` for audio models for tts and asr tasks.
