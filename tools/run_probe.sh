#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
reference_root="${QWEN3_TTS_REFERENCE_ROOT:-/home/fbozzo/projects/Qwen3-TTS}"
model_root="${QWEN3_TTS_MODEL_ROOT:-/home/fbozzo/projects/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
python="${QWEN3_TTS_PYTHON:-$reference_root/.venv/bin/python}"
dependency_root="${QWEN3_TTS_DEPENDENCY_ROOT:-$reference_root/.venv/lib/python3.13/site-packages}"
device="${QWEN3_TTS_DEVICE:-cpu}"

exec "$python" \
  "$repo_root/tests/models/qwen3_tts/reference/probe_official.py" \
  --reference-root "$reference_root" \
  --dependency-root "$dependency_root" \
  --model "$model_root" \
  --out "$repo_root/artifacts/qwen3_tts/internals" \
  --device "$device"
