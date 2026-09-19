# Flash-Next MTP qualification

The predictor uses one RMSNorm over all 10,240 hidden values, a shared
2,560-wide hidden projection per HC branch, and one embedding projection
broadcast to the four branches. It embeds shifted text token IDs; visual
information comes from the target hidden stream and mRoPE positions.

Source pins:

- [vLLM AMD predictor, 751f6807](https://github.com/vllm-project/vllm/blob/751f6807d9cb3de50c27a5f27188c4fb04fe0e2b/vllm/models/qwen4_exp/amd/mtp.py):
  full-width normalization, separate projections, recursive pre-mixer carry
  and text-only inputs.
- [vLLM proposer, same revision](https://github.com/vllm-project/vllm/blob/751f6807d9cb3de50c27a5f27188c4fb04fe0e2b/vllm/v1/spec_decode/llm_base_proposer.py):
  shifts token IDs while retaining target positions; disables external
  multimodal embeddings for draft classes that do not support them.
- [SGLang predictor, 993d1fcc](https://github.com/sgl-project/sglang/blob/993d1fccbaafe3e79d91567d2fc1d665cc94fa50/python/sglang/srt/models/qwen4_exp_mtp.py):
  the same normalization/fusion; also permits supplied multimodal embeddings.
  Gufo follows the text-only vLLM input path.
- Official checkpoint `Qwen/Qwen3.8-Flash-Next`,
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`: its safetensors index contains
  `mtp.fc_embedding`, `mtp.fc_hidden`, `mtp.pre_fc_norm_embedding` and
  `mtp.pre_fc_norm_hidden`.

The GGUF stores `[fc_embedding | fc_hidden]` in each projection row. Loading
splits those encoded rows without dequantization or requantization.
The target's original Q8 output head scores every vocabulary row. There is no
private Q4 head or fixed vocabulary subset. Greedy proposals use full-head
argmax. Sampled proposals use its exact top 64 logits and Gufo's bounded
proposal policy; this does **not** claim upstream draft-sampler equivalence.
Full-vocabulary target verification uses the same FP64 filtered distribution
as AR, with p/q acceptance and residual correction. Boundary tests cover
top-p/min-p support, acceptance, residual draws and seeded replay. Target RNG
draws retain 53 bits; compact proposal masses remain exact multiples of 2^-24.

## Maintained checks

```sh
nix develop -c cmake --build --preset gpu-test \
  --target qwen38_flash_next_gpu_probe
nix develop -c build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_gpu_probe \
  --model "$MODEL" --mtp-model "$MTP" --mtp-audit
```

The audit uses scalar CPU operators and the original GGUF weights, independently
of device repacking and HIP kernels. It checks:

- Exact encoded weight bytes after splitting; adversarial unequal HC scales
  distinguish full-width RMSNorm from branch-wise normalization.
- Normalized hidden input, fusion, attention, MoE residual and head mixer,
  with independently computed attention caches and recursive input carry.
- Text and image-pad IDs, image mRoPE positions and continuation beyond the
  image. An attached image must not cause predictor-side vision encoding.
- Independent batched versus serial predictor bodies and heads, including
  recursive chains. Candidate IDs/scores match host sorting of the full Q8
  head; operator tests separately check Q8 dots against FP64.

Stage comparisons supply the same recorded input to each CPU/GPU stage and
emulate Q8 activation/F16 cache storage on the CPU. This separates operation
errors from accumulated quantization and MoE routing changes. The scalar
`ReferenceModel::MtpStep` also supports uninterrupted float32 execution;
the stage audit is not an end-to-end float32 equivalence claim.

Norm tolerance is 2e-6 relative RMS; fusion/attention 0.002; MoE/head 5e-5.
The initial eight-state text/image audit passed, with maximum fusion/attention
relative RMS below 0.0008. Full-model session tests additionally cover
C2/C4/C6/C8, ragged budgets, sampled acceptance/rejection and cache restoration;
selector/attention operator tests cover sparse deep contexts.

**Remaining limit:** these checks use converted GGUF weights. They do not
establish unquantized-checkpoint equivalence or detect every conversion error.
Pinned Transformers ignores the MTP weights, so its trunk forward is not an
MTP oracle.
