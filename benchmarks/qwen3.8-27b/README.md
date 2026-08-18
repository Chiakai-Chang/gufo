# Qwen3.8-27B BF16 on Strix Halo

Status: 2026-08-18. Native BF16 GGUF bring-up for the CPU and ROCm/HIP
executors on AMD Strix Halo (`gfx1151`).

## Model

- Repository: `unsloth/Qwen3.8-27B-GGUF`
- Revision: `f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`
- Artifact: `BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf`
- Format: split GGUF v3, BF16 weights with F32 normalization and recurrence
  tensors
- Shards: 2
- Total size: 54,657,735,616 bytes (50.90 GiB)
- Tensor parameters: 27.32 billion
- Main tensors: 866
- Main transformer layers: 64
- MTP/next-token prediction layers: 1, excluded from normal forward execution

The model is downloaded directly on the Strix Halo host:

```sh
cd /home/fbozzo/projects/strix-halo.cpp
nix develop -c hf download unsloth/Qwen3.8-27B-GGUF \
  --include 'BF16/*' \
  --local-dir models/Qwen3.8-27B-GGUF \
  --max-workers 8
```

| Shard | Bytes | SHA-256 |
| --- | ---: | --- |
| `00001-of-00002` | 49,986,159,616 | `b9966e82b7a4d87028b5eae061d578ee826305ebf8baea5bfc6e09bad0ba191f` |
| `00002-of-00002` | 4,671,576,000 | `92e3943c4f9bd6292a7bef82369f65fed9bfed088b9df0fb2fa2ce17c9edfa02` |

## Architecture

| Field | Value |
| --- | ---: |
| Hidden size | 5120 |
| FFN size | 17408 |
| Main layers | 64 |
| Full-attention interval | 4 |
| Attention heads | 24 |
| KV heads | 4 |
| Head dimension | 256 |
| SSM QKV width | 10240 |
| SSM recurrent width | 6144 |
| SSM value heads | 48 |
| SSM key heads | 16 |
| SSM state/value dimension | 128 |
| Context length | 262144 |
| Vocabulary | 248320 |

## Runtime

The GGUF reader discovers all shards when either shard path is supplied.
Tensor pointers retain their source-shard identity, allowing the HIP executor
to map each file independently.

On the integrated GPU, `STRIX_GPU_WEIGHT_MODE=auto` prefers mapped, read-only
host pages so the 54.7 GB model is not duplicated in unified memory. The
explicit alternatives are:

```sh
STRIX_GPU_WEIGHT_MODE=mapped ./result/bin/strix-server prompt ...
STRIX_GPU_WEIGHT_MODE=copy ./result/bin/strix-server prompt ...
```

The prompt, chat, and benchmark commands print `[Model Load]` after the GGUF
reader, tokenizer, weights, GPU mapping, caches, and execution arenas are
ready.

## Reproduction

```sh
git add .
nix build

MODEL=models/Qwen3.8-27B-GGUF/BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf

./result/bin/strix-server prompt \
  --model "$MODEL" \
  --max-tokens 8 \
  --verbose \
  "Write one sentence about unified memory."

./result/bin/strix-server prompt \
  --cpu \
  --raw \
  --model "$MODEL" \
  --max-tokens 1 \
  --verbose \
  Hello

./result/bin/strix-server bench \
  --model "$MODEL" \
  --n-prompt 32 \
  --repetitions 1

llama-bench \
  --model "$MODEL" \
  --n-prompt 1,8,32,64,128,256,512 \
  --n-gen 0 \
  --repetitions 3 \
  --n-gpu-layers 99 \
  --flash-attn auto \
  --batch-size 512 \
  --ubatch-size 512 \
  --threads 32 \
  --load-mode mmap

llama-bench \
  --model "$MODEL" \
  --n-prompt 0 \
  --n-gen 8 \
  --repetitions 3 \
  --n-gpu-layers 99 \
  --flash-attn auto \
  --batch-size 512 \
  --ubatch-size 512 \
  --threads 32 \
  --load-mode mmap
```

## Validation

The focused Nix/HIP suite covers split-shard discovery, live 27B metadata,
dynamic CPU state sizing, the 27B SSM layout, batched-vs-sequential HIP
recurrence, attention, projection, RoPE, and normalization equivalence.

## Results

Machine: Ryzen AI MAX+ 395, Radeon 8060S `gfx1151`, 128 GiB unified LPDDR5X,
ROCm 7.2.3. All runs use the default BF16 model without custom quantization.

| Path | Test | Result |
| --- | --- | ---: |
| CPU, 32 OpenMP threads | model load | 0.252 s |
| CPU, 32 OpenMP threads | raw one-token prompt + one generated token | 0.48-0.65 tok/s |
| HIP mapped weights | model load | 1.63-1.95 s |
| HIP mapped weights | raw one-token prompt + one generated token, cold | 0.670 tok/s |
| HIP mapped weights | `tg8`, three repetitions, baseline | 4.17 tok/s |
| HIP mapped weights | `tg8`, three repetitions, optimized | 4.29 tok/s |

The retained HIP change uses 256 threads for the generic BF16 GEMV and fused
FFN gate/up GEMV kernels. SSM and full-attention projection experiments at 256
threads were neutral and were reverted. The measured decode improvement is
2.9%; the remaining BF16 path is bandwidth-bound at roughly the machine's
measured unified-memory copy ceiling.

The llama.cpp comparison uses system `llama-bench` build 10173, commit
`e9fa078`, with the ROCm backend, full GPU offload, automatic flash attention,
and the same split BF16 GGUF. Both engines use three measured repetitions:

| Test | Strix HIP | llama.cpp ROCm | llama.cpp / Strix |
| --- | ---: | ---: | ---: |
| `pp1` | 0.80 +/- 0.02 tok/s | 3.87 +/- 0.17 tok/s | 4.84x |
| `pp8` | 5.56 +/- 0.01 tok/s | 28.20 +/- 0.41 tok/s | 5.07x |
| `pp32` | 20.16 +/- 0.01 tok/s | 74.90 +/- 1.18 tok/s | 3.72x |
| `pp64` | 28.57 +/- 0.06 tok/s | 111.40 +/- 1.36 tok/s | 3.90x |
| `pp128` | 41.71 +/- 0.11 tok/s | 221.03 +/- 2.05 tok/s | 5.30x |
| `pp256` | 48.77 +/- 0.35 tok/s | 242.99 +/- 1.00 tok/s | 4.98x |
| `pp512` | 46.32 +/- 0.18 tok/s | 390.96 +/- 0.85 tok/s | 8.44x |
| `tg8` | 4.29 +/- 0.00 tok/s | 4.02 +/- 0.04 tok/s | 0.94x |

The native HIP prompt path peaks at batch 256. Its 512-token result is lower
because the causal full-attention layers add quadratic work while the GEMMs
are already large enough to saturate the GPU.

llama.cpp is 3.7-8.4x faster for prompt processing, with its advantage growing
at 512 tokens. The native HIP executor is 6.6% faster for steady `tg8` decode.

The `pp32` profile before the decode launch tuning was:

| Stage | Time |
| --- | ---: |
| Total | 1378.2 ms |
| Norms | 1.4 ms |
| Input projections | 309.5 ms |
| SSM recurrence | 36.8 ms |
| SSM output | 56.1 ms |
| FFN GEMMs | 974.4 ms |
