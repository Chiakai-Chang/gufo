# Qwen3-TTS (CustomVoice) — strix port

Status: **IMPLEMENTED (native HIP eager baseline)**

## Upstream sources and provenance

- Official implementation: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS),
  pinned for exactness tests at
  [`022e286b98fbec7e1e916cb940cdf532cd9f488e`](https://github.com/QwenLM/Qwen3-TTS/commit/022e286b98fbec7e1e916cb940cdf532cd9f488e)
  (Apache-2.0).
- The upstream optional Gradio demo selects
  [Source Sans Pro](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/cli/demo.py#L272)
  from Google Fonts, with Arial and generic sans-serif fallbacks. The native
  server does not embed, download, or redistribute those fonts. Source Sans is
  maintained by [Adobe Fonts](https://github.com/adobe-fonts/source-sans) under
  the SIL Open Font License 1.1.
- Official model collection and selected artifact:
  [Qwen3-TTS collection](https://huggingface.co/collections/Qwen/qwen3-tts)
  and
  [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice).
- Architecture reference: [Qwen3-TTS Technical Report,
  arXiv:2601.15621](https://arxiv.org/abs/2601.15621).
- Native implementation ideas and parity methodology were inspected in
  [0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp) at
  [`54ea3d11ba4726f15d072b23a9244dc623f7cfc7`](https://github.com/0xShug0/audio.cpp/commit/54ea3d11ba4726f15d072b23a9244dc623f7cfc7)
  (Apache-2.0). The strix implementation remains model-private under this
  directory and is independently validated against the official Qwen oracle.
- The optional long-form listening fixture is fetched in bounded form from the
  [user-provided SDSU text source](https://dgoldberg.sdsu.edu/515/harrypotter.txt);
  the source text is not vendored in this repository.

## Offline reference implementation

`/home/fbozzo/projects/Qwen3-TTS` (official Alibaba repo, `qwen_tts` package)
is the ground truth. Accuracy is validated against deterministic artifacts
dumped from that repo:

| Artifact | Contents | Determinism |
|---|---|---|
| `talker_codes.npy` | full-sentence codec book, sampled, seed 42 | torch seed |
| `talker_codes_greedy.npy` | bounded five-frame codec book, argmax | fully deterministic |
| `waveform.npy` / `waveform_greedy.npy` | 24 kHz float32 audio | derived |
| `internals/*.npy` | prefill embeddings, layer boundaries, logits | fully deterministic |
| `reference_contract.json` | pinned hashes, shapes, provenance | checked in |

Regenerate with:

```sh
nix develop --command tools/run_ref_tts.sh           # sampled quality, seed 42
nix develop --command tools/run_ref_tts.sh greedy    # five-frame exactness
```

Artifacts land in `artifacts/qwen3_tts/` (gitignored). Weights:
`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (main `model.safetensors` + `speech_tokenizer/model.safetensors`).

An opt-in long-form listening fixture fetches and retains a bounded excerpt
from the user-provided SDSU text source:

```sh
nix develop --command tools/run_ref_tts_quality.sh
```

Its text, codes, and waveform land in `artifacts/qwen3_tts/quality/`; they do
not replace the short exactness contract.

The model is sampling-trained: with both talker samplers disabled it did not
emit EOS before a 3,000-token diagnostic cap. The checked-in greedy contract is
therefore intentionally limited to five generated frames. Full-sentence audio
quality is validated with the seeded sampled artifact instead of treating a
runaway greedy waveform as a quality baseline.

The upstream Python implementation is never used to serve requests. It is an
offline oracle for tensor, logit, codec, waveform, and listening comparisons.
`strix-server` always runs the model-private HIP implementation.

## Architecture (CustomVoice, 12Hz)

`Qwen3TTSForConditionalGeneration`:

* `talker` — 28-layer transformer decoder: hidden 2048, FFN 6144, 16 heads /
  8 KV heads, head_dim 128, RMSNorm eps 1e-6 (per-head q/k RMSNorm too),
  3D interleaved RoPE `mrope_section=[24,20,20]`, theta 1e6, GQA.
  `codec_embedding` 3072→2048, `text_embedding` 151936→2048,
  `text_projection` MLP 2048→2048→2048 (silu), `codec_head` → vocab 3072.
* `code_predictor` (MTP) — 5 layers hidden 1024, 16/8 heads, vocab 2048;
  predicts codebook slots 1..15 after talker head for slot 0.
* speech tokenizer v2 (12Hz) — RVQ decoder + transformer decoder + conv upsample.

## Milestones

1. [DONE] Reference baseline + goldens + exact verification harness
2. [DONE] Loader: safetensors → host tensors + config parse
3. [DONE] Native HIP talker prefill and cached decode
4. [DONE] Native HIP code predictor and checkpoint sampling policy
5. [DONE] Native HIP speech-tokenizer decoder
6. [DONE] HIP-only `strix-server` `/v1/audio/speech` endpoint
7. [DONE] exact and quality tests in `tests/models/qwen3_tts`

The API defaults to the checkpoint's quality policy: top-k 50 sampling at
temperature 0.9 for both the talker and its code predictor, plus repetition
penalty 1.05. `greedy: true` is retained only for bounded deterministic
diagnostics. Sampling is seeded and reproducible within the native backend;
quality parity is judged from upstream tensor/logit agreement and decoded
audio because PyTorch and the native C++ sampler use different random-number
generators.

## gfx1151 validation snapshot

For the required sentence, speaker `vivian`, English, seed 42:

- Talker prompt cosine: `1.0`; prefill-logit cosine: `0.999935`; cached-logit
  cosine: `0.999977`.
- The native speech decoder matches the official ROCm waveform with mean
  absolute error `1.33e-7` and maximum absolute error `1.28e-6`.
- Native sampled audio contains 243 codec frames / 19.44 seconds and completes
  through the HIP-only server in 31.69 seconds.
- The official ROCm safetensors implementation contains 273 frames / 21.84
  seconds and takes 98.00 seconds after model load. Native wall time is 3.09x
  lower and generated-audio throughput is 2.75x higher.
- A shared Whisper tiny.en intelligibility check reports 1.96% WER for native
  audio versus 7.84% for the upstream sampled artifact, with 96.83% compact
  transcript LCS agreement.

Sampled codec frames are not expected to be byte-identical because PyTorch
ROCm and the native backend use different random-number generators. Stable
implementation boundaries are guarded by exact tokens, tensor/logit metrics,
near-exact decoder waveform comparison, fixed-seed native reproducibility, and
the end-to-end intelligibility gate.

### Decode optimization

The production path retains hipBLAS/rocBLAS BF16 GEMM for every content-bearing
projection. A model-private wave32 GEMV was faster, but changed sampled speech
and regressed the strict Whisper intelligibility gate, so it is not retained.

The permanent gfx1151 optimizations preserve the frozen tensors, logits, codec
tokens, and decoded waveform:

- redundant standalone BF16 round launches are removed when the following
  consumer already performs the same conversion;
- q/k input rounding, per-head RMSNorm, and output rounding are fused in a
  128-thread workgroup sized for the model's 128-element heads;
- q/k RoPE output rounding and V rounding share one model-private launch;
- top-k sampling partially sorts only the requested candidates while preserving
  the comparator, candidate order, probability distribution, and RNG sequence;
- safetensor weights are registered and mapped directly on integrated gfx1151
  unified memory instead of being copied or explicitly prefetched.

The retained launch fusions remove 15,141 GPU dispatches per canonical run.
Warm release A/B measurements reduced the 243-frame Dursley workload from
33.03 seconds to 31.69 seconds (4.1%) without quality drift. Experimental
hipBLASLt, direct rocBLAS, graph capture, attention, GEMV, and additional fusion
routes were rejected when they were neutral, slower, or failed exactness.

## Key ids (CustomVoice)

```
codec_eos=2150 codec_bos=2149 codec_pad=2148 codec_think=2154 codec_nothink=2155
codec_think_bos=2156 codec_think_eos=2157
tts_bos=151672 tts_eos=151673 tts_pad=151671 im_start=151644 im_end=151645
spk_id: serena=3066, vivian=3065, uncle_fu=3010, ryan=3061, aiden=2861,
        ono_anna=2873, sohee=2864, eric=2875, dylan=2878
lang codec ids: english=2050 chinese=2055 ... (see config.json)
```
