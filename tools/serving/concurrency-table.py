#!/usr/bin/env python3
"""Render the README concurrency table from serving benchmark artifacts.

Usage:
  concurrency-table.py --dir docs/models/qwen3.8-27b/artifacts

Reads `{gufo-ar,gufo-dflash2,llama-server}-{mixed,repetition}.json` and prints
a Markdown table with end-to-end aggregate output tokens per second per
concurrency level as `mixed / repetition` pairs, exact-match counts against the
Gufo AR C=1 reference, DFlash2 acceptance, DFlash2 over Gufo AR, and Gufo AR
over llama.cpp (the exact-match column says how comparable the outputs are).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SERVERS = ("gufo-ar", "gufo-dflash2", "llama-server")
GROUPS = ("mixed", "repetition")


def _load(directory: Path, name: str, group: str) -> dict[str, Any] | None:
    path = directory / f"{name}-{group}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _result(
    reports: dict[tuple[str, str], dict[str, Any] | None],
    name: str,
    group: str,
    level: int,
) -> dict[str, Any] | None:
    report = reports[(name, group)]
    if report is None:
        return None
    return report["results"].get(f"c{level}")


def _rate(result: dict[str, Any] | None) -> float | None:
    if result is None:
        return None
    return result["aggregate"]["output_tokens_per_second"]["overall"]


def _matched(result: dict[str, Any] | None) -> str:
    if result is None:
        return "-"
    exactness = result.get("completionExactness")
    if not isinstance(exactness, dict):
        return "-"
    return f"{exactness['exactRequests']}/{exactness['comparedRequests']}"


def _acceptance(result: dict[str, Any] | None) -> str:
    if result is None:
        return "-"
    value = result["speculative"]["acceptance"]
    return "-" if value is None else f"{value * 100:.1f}%"


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _ratio(numerator: float | None, denominator: float | None) -> str:
    if numerator is None or denominator is None or denominator <= 0.0:
        return "-"
    return f"{numerator / denominator:.2f}x"


def _pair(values: list[str]) -> str:
    return " / ".join(values)


def render(directory: Path) -> str:
    reports = {
        (name, group): _load(directory, name, group)
        for name in SERVERS
        for group in GROUPS
    }
    levels: list[int] = []
    for report in reports.values():
        if report is not None:
            for key in report["results"]:
                level = int(key[1:])
                if level not in levels:
                    levels.append(level)
    levels.sort()

    header = (
        "| Concurrency | Gufo AR | Gufo DFlash2 | DFlash2 acceptance | "
        "DFlash2 / Gufo AR | llama.cpp AR | llama.cpp exact | "
        "Gufo AR / llama.cpp |"
    )
    rule = (
        "| ----------: | ------: | -----------: | -----------------: | "
        "----------------: | -----------: | --------------: | "
        "------------------: |"
    )
    lines = [header, rule]
    for level in levels:
        ar = [_result(reports, "gufo-ar", group, level) for group in GROUPS]
        dflash = [
            _result(reports, "gufo-dflash2", group, level) for group in GROUPS
        ]
        llama = [
            _result(reports, "llama-server", group, level) for group in GROUPS
        ]
        cells = [
            str(level),
            _pair([_fmt(_rate(result)) for result in ar]),
            _pair([_fmt(_rate(result)) for result in dflash]),
            _pair([_acceptance(result) for result in dflash]),
            _pair(
                [
                    _ratio(_rate(candidate), _rate(baseline))
                    for candidate, baseline in zip(dflash, ar)
                ]
            ),
            _pair([_fmt(_rate(result)) for result in llama]),
            _pair([_matched(result) for result in llama]),
            _pair(
                [
                    _ratio(_rate(gufo), _rate(candidate))
                    for candidate, gufo in zip(llama, ar)
                ]
            ),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(render(args.dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
