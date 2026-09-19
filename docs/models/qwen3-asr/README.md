# Qwen3-ASR 1.7B

Native BF16 safetensors speech recognition on gfx1151. The 1.7B checkpoint is
supported; 0.6B, forced alignment and audio GGUFs are unsupported. Python runs
only offline reference checks. Generation is deterministic greedy decoding.

[Benchmarks](BENCHMARKS.md) · [Evaluation](EVALUATION.md) · [Experiments](EXPERIMENTS.md)

```sh
nix develop -c hf download Qwen/Qwen3-ASR-1.7B \
  --revision 7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --local-dir models/Qwen3-ASR-1.7B
nix build
./result/bin/gufo transcribe --model models/Qwen3-ASR-1.7B --audio speech.wav
./result/bin/gufo serve audio --asr-model models/Qwen3-ASR-1.7B
```

```sh
curl -sS http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@speech.wav -F model=qwen3-asr-1.7b -F response_format=json
```

Inputs: PCM16/24/32 or float32 WAV, up to eight channels and 192 kHz. The
frontend mixes to mono and uses windowed-sinc resampling to 16 kHz.
HTTP supports `json`, `text`, `verbose_json`, optional `language` and `prompt`,
and temperature zero. Streaming and word timestamps are unsupported.

Audio work is serialized; additional concurrent requests queue. Load TTS in the
same audio server with `--tts-model` when needed. See
[server limits](../../SERVER.md) and [quality gaps](EVALUATION.md).
