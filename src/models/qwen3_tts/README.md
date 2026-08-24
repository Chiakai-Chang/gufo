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

- Talker prompt cosine: `1.0`; prefill-logit cosine: `0.999938`; cached-logit
  cosine: `0.99997`; both greedy argmax boundaries match the frozen reference.
- The native speech decoder matches the official ROCm waveform with mean
  absolute error `1.33e-7` and maximum absolute error `1.28e-6`.
- Decode costs `32.4` ms per codec frame, measured on a fixed 200-frame greedy
  request. The canonical sampled request returns 289 frames / 23.12 seconds of
  audio in `9.46` seconds, which is `2.44x` faster than playback.
- The official ROCm safetensors implementation contains 273 frames / 21.84
  seconds and takes 98.00 seconds after model load. Native generated-audio
  throughput is `10.9x` higher.
- A shared Whisper tiny.en intelligibility check reports 1.96% WER for native
  audio and 100% compact transcript LCS against the retained release baseline.

Sampled codec frames are not expected to be byte-identical because PyTorch
ROCm and the native backend use different random-number generators. Stable
implementation boundaries are guarded by exact tokens, tensor/logit metrics,
near-exact decoder waveform comparison, and the end-to-end intelligibility gate.

Fixed-seed identity holds for the bounded eight-frame test but not across a
full-length request: rocBLAS prefill leaves enough run-to-run variation that a
sampled draw eventually diverges, so a full request has no stable byte hash.
This predates the decode work and applies equally to the previous release, so
optimization A/B tests compare per-frame cost and the intelligibility report
rather than output hashes.

### Decode optimization

Decode is bound by DRAM bandwidth, not arithmetic. Each codec frame streams
about `5.2` GB of BF16 weights: `2.7` GB for one pass over the 28-layer talker
and `2.5` GB because the 15 code-predictor heads are sequentially dependent and
each re-reads all five predictor layers. A pure streaming read reaches
`242` GB/s on this part, so that traffic sets a floor near `21` ms per frame.

hipBLAS reached only 40-60 GB/s on these projections: decode presents a single
token, and the batched kernel pads that one column into a 128x32 macro tile with
a split-K reduction. The retained optimizations close the gap to the floor:

- a model-private GEMV reduces one output row per workgroup, so eight
  wavefronts each walk the row with an independent coalesced stream. It sustains
  208-242 GB/s across every decode shape, accumulates in float32 with a fixed
  reduction order, and is *more* accurate than the batched path it replaces
  (maximum relative error `1.95e-4` against an fp64 reference, versus `2.35e-4`);
- the GEMV shares one pass over the weights across up to four input columns, so
  the two-token predictor step costs the same memory traffic as a single token;
- weights are copied into device-resident coarse-grained pages instead of
  host-registered ones. gfx1151 shares one physical memory pool, but the
  registered pages do not cache the same way and measured 12% slower. The copy
  is shifted so the safetensors payload lands 16-byte aligned, which is what
  makes the widest packed loads legal; memory-mapped weights fall back to
  8-byte loads and remain supported via `STRIX_QWEN3_TTS_WEIGHT_MODE=mapped`;
- q/k per-head RMSNorm, the q/k rotation and V rounding share one launch, and
  each residual add folds into the RMSNorm that follows it. Both keep every
  intermediate BF16 rounding, so the arithmetic is unchanged;
- top-k sampling partially sorts only the requested candidates while preserving
  the comparator, candidate order, probability distribution, and RNG sequence;
- the repetition-penalty test uses a per-vocabulary flag instead of scanning the
  generated codes, which had made every frame cost more than the one before it.

Warm release A/B measurements reduced per-frame decode from `129.7` ms to
`32.4` ms (`4.0x`). All new kernels report zero scratch, zero register spills,
and 16 waves/SIMD.

HIP graph replay was implemented and measured before being rejected: back-to-back
dispatch on gfx1151 costs about `2` us whether kernels are launched individually
or replayed from a graph (`2.03` us versus `1.82` us in isolation), and capturing
the predictor changed nothing end to end. Reducing launch count by fusing is
what helps. Experimental hipBLASLt, direct rocBLAS, and attention routes were
rejected when they were neutral, slower, or failed exactness.

The remaining budget per frame is roughly `22` ms of weight streaming already at
the bandwidth roofline, plus about `10` ms of dispatch overhead and short-kernel
execution. Going meaningfully faster requires moving fewer bytes, which means
trading weight precision, so it is deliberately not attempted here.

## Key ids (CustomVoice)

```
codec_eos=2150 codec_bos=2149 codec_pad=2148 codec_think=2154 codec_nothink=2155
codec_think_bos=2156 codec_think_eos=2157
tts_bos=151672 tts_eos=151673 tts_pad=151671 im_start=151644 im_end=151645
spk_id: serena=3066, vivian=3065, uncle_fu=3010, ryan=3061, aiden=2861,
        ono_anna=2873, sohee=2864, eric=2875, dylan=2878
lang codec ids: english=2050 chinese=2055 ... (see config.json)
```
