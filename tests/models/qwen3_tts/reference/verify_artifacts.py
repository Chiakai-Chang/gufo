#!/usr/bin/env python3
"""Verify frozen official Qwen3-TTS artifacts against the checked-in contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--rocm-artifacts", type=Path)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_generation(root: Path, contract: dict, suffix: str, label: str) -> None:
    codes_path = root / f"talker_codes{suffix}.npy"
    waveform_path = root / f"waveform{suffix}.npy"
    require(codes_path.is_file(), f"missing {codes_path}")
    require(waveform_path.is_file(), f"missing {waveform_path}")
    require(
        sha256(codes_path) == contract["talker_codes_sha256"],
        f"{label} codec-token artifact hash drifted",
    )
    require(
        sha256(waveform_path) == contract["waveform_sha256"],
        f"{label} waveform artifact hash drifted",
    )

    codes = np.load(codes_path, allow_pickle=False)
    waveform = np.load(waveform_path, allow_pickle=False)
    require(
        list(codes.shape) == [contract["code_steps"], contract["code_groups"]],
        f"{label} codec-token shape drifted",
    )
    require(
        codes[:5, :6].tolist() == contract["first_codes"],
        f"{label} codec-token prefix drifted",
    )
    require(
        waveform.dtype == np.float32 and waveform.size == contract["samples"],
        f"{label} waveform shape or dtype drifted",
    )


def main() -> int:
    args = parse_args()
    if not args.artifacts.is_dir():
        print(f"SKIP: Qwen3-TTS artifacts not found at {args.artifacts}")
        return 77

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    verify_generation(args.artifacts, contract["greedy"], "_greedy", "CPU greedy")
    if args.rocm_artifacts is not None:
        verify_generation(
            args.rocm_artifacts,
            contract["rocm_greedy"],
            "_greedy",
            "ROCm greedy",
        )
        verify_generation(
            args.rocm_artifacts,
            contract["rocm_sampled"],
            "",
            "ROCm sampled",
        )

    prefill = contract["prefill"]
    internals = args.artifacts / "internals"
    prefill_files = {
        "inputs_embeds_sha256": "prefill_inputs_embeds.npy",
        "layer0_output_sha256": "layer0_output.npy",
        "layer27_output_sha256": "layer27_output.npy",
        "logits_sha256": "prefill_logits_lastpos.npy",
        "predictor_codes_sha256": "predictor_first_frame_codes.npy",
        "predictor_logits_sha256": "predictor_first_frame_logits.npy",
    }
    for contract_key, filename in prefill_files.items():
        path = internals / filename
        require(path.is_file(), f"missing {path}")
        require(
            sha256(path) == prefill[contract_key],
            f"prefill artifact hash drifted: {filename}",
        )

    embeds = np.load(internals / "prefill_inputs_embeds.npy", allow_pickle=False)
    logits = np.load(internals / "prefill_logits_lastpos.npy", allow_pickle=False)
    require(
        list(embeds.shape) == [prefill["tokens"], prefill["hidden_size"]],
        "prefill embedding shape drifted",
    )
    require(
        list(logits.shape) == [1, prefill["logits"]],
        "prefill logits shape drifted",
    )
    predictor_codes = np.load(
        internals / "predictor_first_frame_codes.npy", allow_pickle=False
    )
    predictor_logits = np.load(
        internals / "predictor_first_frame_logits.npy", allow_pickle=False
    )
    require(
        list(predictor_codes.shape) == [1, 16],
        "predictor first-frame code shape drifted",
    )
    require(
        list(predictor_logits.shape) == [15, 2048],
        "predictor first-frame logit shape drifted",
    )

    print(
        "PASS: official Qwen3-TTS artifacts match "
        f"{contract['reference_commit']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
