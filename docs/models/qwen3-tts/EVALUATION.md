# Qwen3-TTS evaluation

The upstream Python implementation never serves a request. It is an offline
oracle for tensor, logit, codec, waveform, and listening comparisons, pinned at
commit `022e286b`.

Sampled codec frames are not expected to be byte-identical, because PyTorch ROCm
and the native C++ sampler use different random-number generators. Parity is
therefore judged from exact tokens, tensor and logit metrics, near-exact decoder
waveform comparison, and an end-to-end intelligibility gate.

The intelligibility gate transcribes the native and official waveforms with the
same native HIP Qwen3-ASR-1.7B runtime and scores both against the requested
text. The migration was rerun on CustomVoice: native output matches the
requested text exactly, while the official ROCm waveform has a `3.92%` WER.

| Variant | Native WER | Official WER | Transcript LCS |
|---|---|---|---|
| CustomVoice | `0%` | `3.92%` | `97.29%` |

VoiceDesign and Base ICL retain their tensor, code, and waveform gates below;
their long-form intelligibility fixtures have not yet been rerun with the new
ASR backend.

### Gate metrics

| Variant | Metric | Measured | Gate |
|---|---|---|---|
| All | Talker prompt cosine | `1.0` | `> 0.99` |
| All | Prefill-logit cosine | `0.999938` | `> 0.95` |
| All | Cached-logit cosine | `0.99997` | `> 0.95` |
| All | Greedy argmax boundaries | match | exact |
| All | Predictor-logit cosine | `0.999539` | `> 0.95` |
| All | Decoder waveform MAE / max | `1.33e-7` / `1.28e-6` | `< 1e-5` / `< 1e-4` |
| CustomVoice | Qwen3-ASR WER / transcript LCS | `0%` / `97.29%` | `< 30%` / `> 75%` |
| VoiceDesign | Prompt cosine, semantic greedy tokens | `1.0`, 5/5 | `> 0.9999`, exact |
| Base | Prompt cosine, bounded greedy codes | `1.0`, 80/80 | `> 0.999`, exact |
| Base | Speaker-embedding cosine | `0.999997` | `> 0.9999` |
| Base | Speech-code agreement, aggregate | `0.577351` | `> 0.55` |
| Base | Speech-code agreement, groups 0 / 1 | `0.970` / `0.950` | `> 0.95` / `> 0.90` |
| Base | Speech reconstruction cosine / MAE | `0.939502` / `0.00827` | `> 0.93` / `< 0.02` |

### Known limitations

- **Full-length replay:** identical fixed-seed requests can diverge after a long
  prefix (observed at frame 38 of a 60-frame control), including greedy runs.
  Eight-frame tests and isolated decoder controls repeat; root cause is unknown.
- **Greedy EOS:** a diagnostic reached 3000 frames without EOS. Bound greedy
  checks; use sampled output for complete-sentence quality.
- **Base speech codes:** aggregate agreement 0.577351 is close to the official
  BF16-versus-F32 self-control (0.5514); discrete RVQ decisions amplify rounding.
  This does not justify relaxing the retained waveform/embedding gates.
- **Long reference clips:** the official 250-frame encoder window is wired,
  but the retained 202-frame clip does not exercise it. A longer oracle is TODO.
- VoiceDesign/Base long-form intelligibility needs a current ASR-backed refresh.

## Reference and focused checks

Official QwenLM/Qwen3-TTS revision
`022e286b98fbec7e1e916cb940cdf532cd9f488e` (Apache-2.0). Model fixtures and quality
scripts live in `tests/models/qwen3_tts`; tensor payloads stay in ignored
`artifacts/qwen3_tts`. Full logits are regenerated, not stored as experiment dumps.

```sh
nix develop -c ctest --preset gpu-full -R qwen3_tts --output-on-failure
nix develop -c tools/audio/run_ref_tts.sh greedy
nix develop -c tools/audio/run_ref_tts.sh
nix develop -c python3 tests/models/qwen3_tts/quality/compare_intelligibility.py \
  --reference-npy artifacts/qwen3_tts/rocm/waveform.npy \
  --candidate-wav native.wav --asr-model "$ASR_MODEL" --gufo result/bin/gufo
```

Use five-frame greedy oracles for exact codes, sampled seed-42 oracles for
complete speech, and the same ASR runtime to transcribe both waveforms.
`--contract` supplies variant-specific text. VoiceDesign/Base captures include
prompt and speaker/speech-encoder boundaries. Keep pinned contract hashes,
shapes, dtype and upstream provenance with each generated reference.
