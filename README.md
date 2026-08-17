# Strix-Halo.cpp

Strix-Halo.cpp is a local inference runtime built specifically for AMD Strix
Halo systems with a `gfx1151` RDNA 3.5 GPU, an XDNA2 NPU, and up to 128 GiB of
unified memory.

The project is intentionally not a general-purpose inference framework. It
will support only the best available open-weights that can run comfortably on
the hardware.

Model-specific quantization, kernels, graph structure, scheduling policy, and
memory layout will be extremely tailored for Strix Halo.

## Current Status

The native C++ executable is currently a HIP/XRT hardware probe, not yet an
inference runtime. The implemented model work is the offline Python
quantization and quality toolchain under `tools/`, including Qwen3.5-0.8B
SHQ4/SHQ6/SHQ8 experiments.

Qwen3.5-0.8B is the rapid-iteration model. Qwen3.8-27B is the first production
model. The first production artifact and runtime capability are text-only; the
vision encoder is excluded.

The first native inference milestone is deterministic GPU-only greedy text
generation from a terminal prompt. GPU/NPU interoperability and NPU prefill
remain evidence-gated parallel work.

See:

- [Project status and decisions](docs/PROJECT_STATUS.md)
- [Implementation roadmap](docs/ROADMAP.md)
- [Offline tools](tools/README.md)
- [Qwen3.5-0.8B benchmark](benchmarks/qwen3.5-0.8b/README.md)

## Supported Platform

Linux x86-64 on AMD Strix Halo (`gfx1151` GPU and XDNA2 NPU) is the only
planned production platform. Windows, macOS, and CUDA are out of scope.

Build through Nix:

```sh
nix build
./result/bin/strix
```

## Reference Projects

The initial design is informed by the following open source projects:

- `llama.cpp` for compact model serving, GGUF, and CPU/GPU correctness paths.
- `vLLM` for continuous batching and paged request scheduling.
- `hipEngine` for torch-free HIP execution, and native speculative-cycle work.
- `ROCmFPX` for activation-aware quantization and quality evaluation.
- `DwarfStar` for DeepSeek V4 Flash, MoE scheduling, DSpark.
- `ypapadop-amd/ggml` `hsa-backend` for XDNA2 HSA dispatch and MLIR-AIE
  integration patterns.
