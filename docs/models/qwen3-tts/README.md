# Qwen3-TTS 12Hz 1.7B

Native gfx1151 speech synthesis from BF16 safetensors, with an F32 waveform
decoder. Checkpoint configuration selects the variant; 0.6B, 25Hz and GGUF
checkpoints are unsupported. Official Python is an offline oracle only.

[Benchmarks](BENCHMARKS.md) · [Evaluation](EVALUATION.md) · [Experiments](EXPERIMENTS.md)

## Modes

| Checkpoint suffix | HTTP model ID | Conditioning |
| --- | --- | --- |
| `CustomVoice` | `qwen3-tts-12hz-1.7b-customvoice` | `voice`: checkpoint speaker name. |
| `VoiceDesign` | `qwen3-tts-12hz-1.7b-voice-design` | `instruct`: voice description. |
| `Base` | `qwen3-tts-12hz-1.7b-base` | Reference audio and transcript (ICL), or `voice_clone_mode: "speaker_embedding_only"`. |

```sh
nix develop -c hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --local-dir models/Qwen3-TTS-12Hz-1.7B-Base
nix build
./result/bin/gufo serve audio \
  --tts-model models/Qwen3-TTS-12Hz-1.7B-Base \
  --voice narrator=reference.wav --voice-text narrator=reference.txt
```

Substitute `CustomVoice` or `VoiceDesign` in the download and model directory
for those variants; reference-voice options apply to Base. `--voice NAME=WAV`
registers a reusable clone. `--voice-text` supplies a transcript (text or file),
otherwise a sibling `.txt` is used; absent text selects speaker-only cloning.
`--voice-lang NAME=LANGUAGE` sets that voice's default language.

```sh
curl -sS http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-tts-12hz-1.7b-base","voice":"narrator","input":"Hello from Gufo."}' \
  --output speech.wav
```

`GET /v1/audio/voices` lists usable voices. Base also accepts per-request
`reference_audio` and `reference_text`; see [the audio API](../../SERVER.md#other-model-services)
for formats, limits and language/sampling controls. Both audio services can
share a process by adding `--asr-model`.

Default sampling follows the checkpoint: top-k 50, temperature 0.9 for talker
and predictor, repetition penalty 1.05. Greedy is a bounded diagnostic and
can fail to terminate naturally. Full-length fixed-seed waveform replay is
also an open issue; read [quality limits](EVALUATION.md#known-limitations).
Requests are serialized and return complete audio; streaming is unsupported.
