#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
reference_root="${QWEN3_TTS_REFERENCE_ROOT:-/home/fbozzo/projects/Qwen3-TTS}"
model_root="${QWEN3_TTS_MODEL_ROOT:-/home/fbozzo/projects/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
official_python="${QWEN3_TTS_PYTHON:-$reference_root/.venv/bin/python}"
output_root="$repo_root/artifacts/qwen3_tts/quality"
text_path="$output_root/long_form.txt"
mode="${1:-sampled}"
extra=()
if [[ "$mode" == "greedy" ]]; then
  extra+=(--greedy)
elif [[ "$mode" != "sampled" ]]; then
  echo "usage: tools/audio/run_ref_tts_quality.sh [greedy|sampled]" >&2
  exit 2
fi
if [[ -n "${QWEN3_TTS_MAX_NEW_TOKENS:-}" ]]; then
  extra+=(--max-new-tokens "$QWEN3_TTS_MAX_NEW_TOKENS")
fi

python \
  "$repo_root/tests/models/qwen3_tts/quality/fetch_long_form.py" \
  --output "$text_path"

exec "$official_python" \
  "$repo_root/tests/models/qwen3_tts/reference/generate_artifacts.py" \
  --reference-root "$reference_root" \
  --model "$model_root" \
  --out "$output_root" \
  --text-file "$text_path" \
  "${extra[@]}"
