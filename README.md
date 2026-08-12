# Strix-Halo.cpp

Strix-Halo.cpp is a local inference runtime built specifically for AMD Strix
Halo systems with a `gfx1151` RDNA 3.5 GPU, an XDNA2 NPU, and up to 128 GiB of
unified memory.

The project is intentionally not a general-purpose inference framework. It
will support only the best available open-weights that can run comfortably on
the hardware.

Model-specific quantization, kernels, graph structure, scheduling policy, and
memory layout will be extremely tailored for Strix Halo.

## Reference Projects

The initial design is informed by the following open source projects:

- `llama.cpp` for compact model serving, GGUF, and CPU/GPU correctness paths.
- `vLLM` for continuous batching and paged request scheduling.
- `hipEngine` for torch-free HIP execution, and native speculative-cycle work.
- `ROCmFPX` for activation-aware quantization and quality evaluation.
- `DwarfStar` for DeepSeek V4 Flash, MoE scheduling, DSpark.
- `ypapadop-amd/ggml` `hsa-backend` for XDNA2 HSA dispatch and MLIR-AIE
  integration patterns.
