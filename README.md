# Gufo: a Strix Halo inference engine

Gufo is a vertical local inference engine specifically built and optimized for the AMD Strix Halo hardware:
Ryzen AI MAX+ 395 systems with Radeon 8060S (`gfx1151`), an XDNA2 NPU, and up to 128 GiB of unified memory.

Supported models:

- antirez's [DeepSeek-V4-Flash IQ2XXS](https://huggingface.co/antirez/deepseek-v4-gguf/blob/main/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf) GGUF with [DSpark](https://huggingface.co/antirez/deepseek-v4-gguf/blob/main/DeepSeek-V4-Flash-DSpark-support-0731.gguf) support.
  With its MoE architecture 284B parameters (13B active) and quantization-aware training techniques, it is the largest and smartest text model that this hardware without losing too much of its full-quality accuracy.
- [Qwen3.8-27B:UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-UD-Q4_K_XL.gguf) and [Qwen3.8-27B-UD-Q8_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-UD-Q8_K_XL.gguf) GGUFs from unsloth with z-lab's [DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF). The best choice when you can't saturate your unified memory and want leave room for something else.
- [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) Safetensors text to video generation model.
  Even though the community has built quicker implementations, as for now we decided to support just MiniMaxAI's official one to retain the full model quality.
- [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) Safetensors for audio to text.
- [Qwen3-TTS](https://huggingface.co/collections/Qwen/qwen3-tts) 1.7B models: Base, CustomVoice, and VoiceDesign.

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

## Development Presets

Reproducible CMake presets are configured in `CMakePresets.json` and must be entered through the Nix development environment (`nix develop`). Direct host CMake is unsupported.

| Preset          | Purpose             | Description                                                                        |
| --------------- | ------------------- | ---------------------------------------------------------------------------------- |
| `development`   | Development         | CPU-only Debug build with warnings (`-Wall -Wextra -Wpedantic`)                    |
| `release`       | Release             | CPU-only optimized Release build                                                   |
| `cpu-sanitizer` | Diagnostics         | CPU-only build with AddressSanitizer (ASan) and UndefinedBehaviorSanitizer (UBSan) |
| `cpu-test`      | Unit Tests          | CPU-only test suite executed with CTest                                            |
| `gpu-test`      | GPU Tests           | ROCm/HIP enabled for `gfx1151` without XRT                                         |
| `hardware-test` | Full Hardware Build | Pinned ROCm/HIP (`gfx1151`) and XRT (`XDNA2`) stacks                               |

### Development Workflow

```sh
# Development build
nix develop -c cmake --preset development
nix develop -c cmake --build --preset development

# Release build
nix develop -c cmake --preset release
nix develop -c cmake --build --preset release

# Sanitizer build & test
nix develop -c cmake --preset cpu-sanitizer
nix develop -c cmake --build --preset cpu-sanitizer
nix develop -c ctest --preset cpu-sanitizer --output-on-failure

# CPU tests
nix develop -c cmake --preset cpu-test
nix develop -c cmake --build --preset cpu-test
nix develop -c ctest --preset cpu-test --output-on-failure

# HIP GPU tests (gfx1151)
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test
nix develop -c ctest --preset gpu-full --output-on-failure

# Complete gfx1151 and XDNA2 tests
nix develop -c cmake --preset hardware-test
nix develop -c cmake --build --preset hardware-test
nix develop -c ctest --preset hardware-full --output-on-failure
```

Hardware presets label tests so unavailable devices skip normally during development. Strict presence validation can be enforced with:

- `GUFO_REQUIRE_HIP=1` (or `GUFO_REQUIRE_GPU=1`)
- `GUFO_REQUIRE_XDNA2=1` (or `GUFO_REQUIRE_NPU=1`)

## Reference Projects

The initial design is informed by the following open source projects:

- `llama.cpp` for compact model serving, GGUF, and CPU/GPU correctness paths.
- `vLLM` for continuous batching and paged request scheduling.
- `hipEngine` for torch-free HIP execution, and native speculative-cycle work.
- `ROCmFPX` for activation-aware quantization and quality evaluation.
- `DS4` for DeepSeek V4 Flash, MoE scheduling, and DSpark.
- `ypapadop-amd/ggml` `hsa-backend` for XDNA2 HSA dispatch and MLIR-AIE
  integration patterns.
