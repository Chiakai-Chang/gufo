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

./result/bin/strix-bench \
  --model "$MODEL" \
  --n-prompt 32,64,128,256,512 \
  --n-gen 0 \
  --repetitions 3

./result/bin/strix-bench \
  --model "$MODEL" \
  --validate-prefill 32 \
  --n-prompt 32 \
  --n-gen 0 \
  --repetitions 1

nix develop -c rocprofv3 \
  --kernel-trace \
  --scratch-memory-trace \
  --stats \
  --summary \
  -- ./result/bin/strix-bench \
    --model "$MODEL" \
    --n-prompt 512 \
    --n-gen 0 \
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

Prompt quality is checked using the unchanged sequential `ForwardToken` path
as the numerical reference. `--validate-prefill` runs the same token sequence
through sequential and batched prefill, copies the complete final-token
vocabulary logits to the host, and requires finite logits and identical top-1
tokens. It also reports maximum and mean absolute error, RMSE, and cosine
similarity.

| Path | Prompt | Top-1 | RMSE | Cosine similarity |
| --- | ---: | ---: | ---: | ---: |
| Original batched path | 32 tokens | 222 | 0.01224522 | 0.99998373 |
| Optimized batched path | 32 tokens | 222 | 0.01086507 | 0.99998587 |
| Optimized batched path | 128 tokens | 194 | 0.01236322 | 0.99998993 |

The top-1 IDs differ between the 32-token and 128-token prompts because their
last input tokens differ. In both cases the optimized batch result matches its
sequential reference. A deterministic end-to-end raw completion after
optimized prefill remains coherent:

```text
The capital of France is Paris. The capital of Germany is Berlin. The capital
of Italy is Rome. The capital of Spain is
```

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
| `pp32` | 79.31 +/- 0.20 tok/s | 74.90 +/- 1.18 tok/s | 0.94x |
| `pp64` | 148.37 +/- 1.02 tok/s | 111.40 +/- 1.36 tok/s | 0.75x |
| `pp128` | 205.19 +/- 1.26 tok/s | 221.03 +/- 2.05 tok/s | 1.08x |
| `pp256` | 257.14 +/- 1.10 tok/s | 242.99 +/- 1.00 tok/s | 0.94x |
| `pp512` | 329.82 +/- 1.04 tok/s | 390.96 +/- 0.85 tok/s | 1.19x |
| `tg8` | 4.29 +/- 0.00 tok/s | 4.02 +/- 0.04 tok/s | 0.94x |

The optimized native path is faster than llama.cpp at `pp32`, `pp64`, and
`pp256`, and remains 7.2% and 15.6% behind at `pp128` and `pp512`,
respectively. The native HIP executor remains 6.6% faster for steady `tg8`
decode.

## Prompt Optimization

The original `rocprofv3` trace showed one rocBLAS GEMM kernel family consuming
82.6% of `pp512` GPU time, with a second GEMM family consuming 10.7%. Exact
Qwen3.8 projection-shape probes showed 4-6x lower kernel time with
hipBLASLt-selected algorithms. The executor now caches a hipBLASLt plan by
batch and matrix shape for the large BF16 projections, while retaining the
hipBLAS path for the small alpha/beta projections.

After moving the projections, `rocprofv3 --scratch-memory-trace` identified
the serial DeltaNet recurrence as the next bottleneck. The old kernel used 192
VGPRs and 988 bytes of scratch per thread. Splitting each 128-element state row
across an adjacent lane pair preserves recurrence order while removing
scratch traffic.

| Metric | Original | Optimized |
| --- | ---: | ---: |
| `pp512` | 46.47 tok/s | 329.82 tok/s |
| DeltaNet recurrence, profiled warmup + `pp512` | ~565 ms | ~100 ms |
| Recurrence VGPRs | 192 | 136 |
| Recurrence scratch per thread | 988 bytes | 0 bytes |
| Recurrence scratch allocation | 40.6 MiB | 0 bytes |

This is a 7.10x `pp512` throughput improvement. In the final trace,
hipBLASLt GEMMs account for about 78.9% of GPU time, attention for 7.6%,
DeltaNet recurrence for 5.1%, and the remaining small rocBLAS projections for
2.3%.
