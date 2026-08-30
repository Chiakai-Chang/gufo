# Qwen3-ASR 1.7B

Status: native production implementation.

This directory contains the model-private Qwen3-ASR-1.7B port for AMD Strix
Halo (`gfx1151`). The production runtime uses native C++/HIP; the official
Python package is retained only as an offline correctness oracle.

The directory is self-contained: it depends on no other model's sources, and
`gufo_qwen3_asr` does not link `gufo_core`. Kernels, the JSON reader, and the BPE
tokenizer that were previously shared with `src/models/qwen` and
`src/models/qwen3_tts` are model-private, so neither model can change this one's
numerics. See the module-independence section of
[benchmarks/qwen3-asr/README.md](../../../benchmarks/qwen3-asr/README.md).

Only `Qwen/Qwen3-ASR-1.7B` is in scope. The 0.6B model and forced aligner are
deliberately rejected.

## Architecture

- Audio tower: 128-bin Whisper log-mel input, three stride-2 2D
  convolutions, projection to 1024, 24 bidirectional Transformer layers with
  16 heads and 4096-wide GELU FFNs, then a 1024-to-2048 projection.
- Text decoder: 28 causal Transformer layers, hidden size 2048, FFN size
  6144, 16 attention heads, 8 KV heads, head dimension 128, q/k RMSNorm,
  interleaved multimodal RoPE, and a 151936-token Qwen2 byte-level BPE.
- Generation: deterministic greedy decode, ending at token 151643 or 151645.

## Official reference

The pinned upstream source revision is
`7c6daf77a2421100f5fb066495372c00129d39ff`. The checkpoint snapshot is
`7278e1e70fe206f11671096ffdd38061171dd6e5`.

`reference/run_official.py` records preprocessing, audio-tower boundaries,
the first text-layer boundary, prefill logits, generated token IDs, and final
transcription. These artifacts are the quality contract for the native port.

On the Strix Halo host:

```sh
nix develop -c /home/fbozzo/projects/Qwen3-ASR-reference/.venv/bin/python \
  src/models/qwen3_asr/reference/run_official.py \
  --reference-root /home/fbozzo/projects/Qwen3-ASR-reference \
  --model /var/llms/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --audio /tmp/qwen3-asr-en.wav \
  --out artifacts/qwen3_asr/official
```

The first official BF16 run transcribed the upstream English fixture in
5.20 seconds after a 1.59-second load. It generated 49 tokens:

> Uh huh. Oh yeah, yeah. He wasn't even that big when I started listening to
> him, but and his solo music didn't do overly well, but he did very well when
> he started writing for other people.

## Native usage

Build the release binary with Nix and transcribe a WAV:

```sh
nix build
./result/bin/gufo transcribe \
  --model /var/llms/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --audio /tmp/qwen3-asr-en.wav
```

The `asr` command is an alias for `transcribe`. Inputs may be PCM16, PCM24,
PCM32, or float32 RIFF WAV, with up to eight interleaved channels and sample
rates through 192 kHz. The runtime mixes to mono and uses a windowed-sinc
resampler for the model's 16 kHz frontend.

An OpenAI-compatible endpoint is also available:

```sh
./result/bin/gufo serve asr \
  --model /var/llms/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5

curl -sS http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@/tmp/qwen3-asr-en.wav \
  -F model=qwen3-asr-1.7b \
  -F response_format=json
```

The endpoint supports `json`, `text`, and `verbose_json`, optional language
and prompt fields, and deterministic temperature zero. Streaming and word
timestamps are not implemented.

## Quality

The native quality gate compares the official and native frontend, audio
encoder, text decoder, prefill logits, every generated token, and final
transcript.

| Boundary | Cosine | Relative L2 | Maximum absolute error |
|---|---:|---:|---:|
| CPU log-mel frontend | `1.0` | `9.43e-7` | `7.21e-5` |
| HIP convolutional frontend | `1.0` | `1.11e-4` | `0.015625` |
| Audio encoder layer 0 | `0.999998` | `0.00199` | `0.0625` |
| Complete audio encoder | `0.999855` | `0.01705` | `0.00830` |
| Text decoder layer 0 | `0.999998` | `0.00203` | `0.0625` |
| Prefill logits | `0.999703` | `0.02444` | `0.63281` |

Greedy generation reproduces all 49 official token IDs and the exact
transcript. The model-private argmax uses a `0.25` BF16 tie window for the
retained decode GEMV route: the official sequence has a `35.5/35.5` tie, while
the alternate accumulation order produces `35.25/35.5`; every captured
non-tied official margin is at least `1.0`.

## Performance

Current numbers, the mechanism behind each one, and the roofline study live in
[benchmarks/qwen3-asr/README.md](../../../benchmarks/qwen3-asr/README.md).
Release measurements use one resident runtime, one warmup, and five measured
runs. The fixture is 15.051 seconds of English audio.

| Route | Median total | Text decoder | Real-time factor | Audio / wall |
|---|---:|---:|---:|---:|
| BF16 hipBLAS decode | `2.472 s` | `2.417 s` | `0.1642` | `6.09x` |
| Unfused batch-one decode GEMV | `1.673 s` | `1.619 s` | `0.1111` | `9.00x` |
| Fused LDS-staged GEMV + native attention | `0.992 s` | `0.942 s` | `0.0659` | **`15.17x`** |

The retained route loads in about `0.60 s` and transcribes 15 seconds of speech
in under a second: `15.2x` faster than real time, and `5.24x` faster than the
official implementation's `5.20 s` generate-only time while additionally doing
WAV decoding, resampling, and feature extraction.

The decode projections are `77.8%` of GPU time and now run at `91-99%` of the
measured `240.24 GB/s` DRAM read ceiling, so they are bandwidth-bound with no
headroom left at BF16. Decode attention is `5.2%`.

Rejected on measurement: a fused gate/up SwiGLU that preserved all tokens but
regressed the request; non-temporal weight loads, which bypass the 32 MB MALL
and cost more than half the GEMV's throughput; and a hand-written blocked BF16
WMMA kernel for the multi-token projections, which is `2.5x` slower than
hipBLASLt at these shapes. No attention path uses FlashAttention, and the
benchmark card explains why porting one is capped at about `1%` here.

Reference fallbacks remain available for A/B work. Each reproduces the same 49
official token IDs:

```sh
GUFO_QWEN3_ASR_TEXT_GEMV=unfused          # one decode GEMV dispatch per tensor
GUFO_QWEN3_ASR_TEXT_GEMV=hipblas          # use hipBLAS for batch-one decode
GUFO_QWEN3_ASR_TEXT_ATTENTION=batched     # use the shared batched attention
GUFO_QWEN3_ASR_TEXT_ATTENTION=hipblas     # use eager per-head BF16 QK/PV
GUFO_QWEN3_ASR_PREFILL_GEMM=hipblas       # use hipBLAS for multi-token GEMM
GUFO_QWEN3_ASR_WEIGHT_MODE=mapped         # map checkpoint pages instead of copying
GUFO_QWEN3_ASR_GEMM_RANK=8                # hipBLASLt algorithm rank: 8 is 11 ms
                                          # faster, marginally less accurate
GUFO_QWEN3_ASR_GEMM_TRACE=1               # dump hipBLASLt plan selection
```

## Tests and tools

```sh
nix build .#checks.x86_64-linux.pr
nix develop -c ctest --preset hardware-full -R qwen3_asr
nix develop -c python3 src/models/qwen3_asr/tools/benchmark.py \
  --model /var/llms/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --audio /tmp/qwen3-asr-en.wav

# Batch-one decode GEMV ablation against the measured DRAM read ceiling.
nix develop -c tools/bench/build.sh asr_decode_gemv_bench
/tmp/asr_decode_gemv_bench
```

Focused unoptimized binaries under `build/gpu-test` are only for correctness
and debugging. Performance numbers must come from `result/bin/gufo`.
