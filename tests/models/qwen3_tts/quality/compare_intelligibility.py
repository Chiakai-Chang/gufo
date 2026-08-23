#!/usr/bin/env python3
"""Compare sampled native Qwen3-TTS audio with the upstream quality oracle."""

from __future__ import annotations

import argparse
import json
import math
import re
import wave
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument("--reference-npy", type=Path)
    reference.add_argument("--reference-wav", type=Path)
    parser.add_argument("--candidate-wav", type=Path, required=True)
    parser.add_argument("--asr-model", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("tests/models/qwen3_tts/reference_contract.json"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-wer", type=float, default=0.30)
    parser.add_argument("--max-wer-regression", type=float, default=0.15)
    parser.add_argument("--min-transcript-lcs", type=float, default=0.75)
    parser.add_argument("--min-duration-ratio", type=float, default=0.75)
    return parser.parse_args()


def read_pcm16_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if channels != 1 or sample_width != 2:
        raise ValueError("candidate must be a mono PCM16 WAV")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return audio, sample_rate


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_word in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_word in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + (left_word != right_word),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(actual: str, expected: str) -> float:
    expected_words = normalize_words(expected)
    if not expected_words:
        raise ValueError("expected transcript is empty")
    return edit_distance(normalize_words(actual), expected_words) / len(expected_words)


def compact_lcs_ratio(left: str, right: str) -> float:
    left_chars = "".join(normalize_words(left))
    right_chars = "".join(normalize_words(right))
    if not left_chars or not right_chars:
        return 0.0
    previous = [0] * (len(right_chars) + 1)
    for left_char in left_chars:
        current = [0]
        for index, right_char in enumerate(right_chars, start=1):
            if left_char == right_char:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1] / max(len(left_chars), len(right_chars))


def resample(audio: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == 16000:
        return audio
    divisor = math.gcd(source_rate, 16000)
    return resample_poly(audio, 16000 // divisor, source_rate // divisor).astype(
        np.float32
    )


def transcribe(
    audio: np.ndarray,
    sample_rate: int,
    processor: AutoProcessor,
    model: AutoModelForSpeechSeq2Seq,
    device: torch.device,
) -> str:
    inputs = processor(
        resample(audio, sample_rate), sampling_rate=16000, return_tensors="pt"
    )
    features = inputs.input_features.to(device=device, dtype=model.dtype)
    with torch.inference_mode():
        generated = model.generate(features)
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def main() -> int:
    args = parse_args()
    with args.contract.open(encoding="utf-8") as handle:
        expected_text = json.load(handle)["text"]
    if args.reference_npy is not None:
        reference = np.load(args.reference_npy).astype(np.float32).reshape(-1)
        reference_rate = 24000
    else:
        reference, reference_rate = read_pcm16_wav(args.reference_wav)
    candidate, candidate_rate = read_pcm16_wav(args.candidate_wav)
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("audio contains non-finite samples")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.asr_model, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.asr_model, dtype=dtype, local_files_only=True
    ).to(device)

    reference_text = transcribe(
        reference, reference_rate, processor, model, device
    )
    candidate_text = transcribe(candidate, candidate_rate, processor, model, device)
    reference_wer = word_error_rate(reference_text, expected_text)
    candidate_wer = word_error_rate(candidate_text, expected_text)
    transcript_lcs = compact_lcs_ratio(reference_text, candidate_text)
    reference_duration = reference.size / reference_rate
    candidate_duration = candidate.size / candidate_rate
    duration_ratio = min(reference_duration, candidate_duration) / max(
        reference_duration, candidate_duration
    )
    failures = []
    if candidate_wer > args.max_wer:
        failures.append("candidate_wer")
    if candidate_wer > reference_wer + args.max_wer_regression:
        failures.append("wer_regression")
    if transcript_lcs < args.min_transcript_lcs:
        failures.append("transcript_lcs")
    if duration_ratio < args.min_duration_ratio:
        failures.append("duration_ratio")

    report = {
        "ok": not failures,
        "failures": failures,
        "device": str(device),
        "reference_transcript": reference_text,
        "candidate_transcript": candidate_text,
        "reference_wer": reference_wer,
        "candidate_wer": candidate_wer,
        "transcript_lcs": transcript_lcs,
        "reference_duration_seconds": reference_duration,
        "candidate_duration_seconds": candidate_duration,
        "duration_ratio": duration_ratio,
        "reference_peak": float(np.max(np.abs(reference))),
        "candidate_peak": float(np.max(np.abs(candidate))),
        "reference_rms": float(np.sqrt(np.mean(np.square(reference)))),
        "candidate_rms": float(np.sqrt(np.mean(np.square(candidate)))),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
