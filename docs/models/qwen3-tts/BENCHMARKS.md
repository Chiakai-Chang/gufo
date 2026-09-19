# Qwen3-TTS benchmarks

Measured on gfx1151 with the release build, warm, for the canonical sentence at
seed 42.

| | Native | Official ROCm | Ratio |
|---|---|---|---|
| Canonical request | `9.05`–`9.16` s for 23.12 s of audio | `98.00` s for 21.84 s | `10.9x` throughput |
| Real-time factor | `2.55x` | `0.22x` | — |
| Per codec frame | `31.0` ms | — | — |

Per-variant endpoint latency:

| Variant | Audio | Request |
|---|---|---|
| CustomVoice | 23.12 s | `9.09` s |
| VoiceDesign | 6.96 s | `2.90` s |
| Base ICL | 5.20 s | `2.69` s, or `2.45` s reusing the reference clip |

## Concurrent HTTP

| C | engine | wall | throughput | latency p50 | latency max |
| --- | --- | --- | --- | --- | --- |
| 1 | gufo | 4.48 s | 2.29 audio-s/s | 4.48 s | 4.48 s |
| 2 | gufo | 8.95 s | 2.29 | 6.72 s | 8.95 s |
| 4 | gufo | 17.90 s | 2.29 | 11.19 s | 17.90 s |
| 8 | gufo | 35.84 s | 2.29 | 20.17 s | 35.84 s |

Concurrent data: 2026-09-10, Base, greedy, warmed HTTP, synchronized bursts.
The server serializes requests. C6 and a current sampled concurrency refresh:
**TODO**. Single-request variant numbers use different texts and cannot be
compared as equivalent workloads.

A separate 2026-09-10 BF16 Base comparison (two warmups, median of three,
English/Italian short/medium texts) averaged RTF **0.472** for Gufo and **0.663**
for audio.cpp `3174e6b`. RTF is wall/audio (lower is better); it is distinct
from audio/wall real-time multiples above. No MOS study was performed.

## Reproduce

Use the [model server examples](README.md), a fixed text/reference voice,
checkpoint identity and seed. Measure resident requests after warmup; report
output audio duration, wall time and RTF. Use sampled requests for natural EOS;
limit codec frames for greedy operator controls. Profiler runs are separate.

Retained profile: talker/predictor GEMV 22.7 ms per frame, waveform decode
2.2 ms, other kernels 3.1 ms, dispatch gaps 3.8 ms. Projection traffic is near
the measured DRAM ceiling; these bounded-profile components are not a new
end-to-end latency measurement.
