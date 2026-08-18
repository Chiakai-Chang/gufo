"""Pure text answer extractors for capability evaluations (MCQ, Integer, LineSpec).

No model execution or runtime dependencies; deterministic regex/string parsing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


def extract_mcq_answer(text: str) -> str | None:
    """Extracts a multiple-choice option letter (A, B, C, D, etc.) from model response text."""
    if not text:
        return None

    # Priority 1: Boxed answer, e.g. \boxed{B} or \boxed{\text{B}}
    boxed_match = re.search(r"\\boxed\{\s*(?:\\text\{)?\s*([A-Ha-h])\s*\}?\}", text)
    if boxed_match:
        return boxed_match.group(1).upper()

    # Priority 2: Explicit prefix, e.g. "Answer: (B)", "The correct option is **C**", "Choice D"
    patterns = [
        r"(?:answer|option|choice)\s*(?:is|:|=)\s*(?:is\s*)?[\*\_\(\[\s]*([A-Ha-h])[\*\_\)\]\s\.\,]",
        r"[\*\_\(]\s*([A-Ha-h])\s*[\*\_\)]\s*(?:is\s+correct|is\s+the\s+answer)",
        r"(?:^|\n)\s*(?:answer|option|choice)\s*:\s*([A-Ha-h])\b",
        r"(?:^|\n)\s*([A-Ha-h])[\.\)]\s",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # Priority 3: Last standalone letter in response if clearly highlighted
    last_letter_match = re.findall(r"\b([A-D])\b", text)
    if last_letter_match:
        return last_letter_match[-1].upper()

    return None


def extract_integer_answer(text: str) -> int | None:
    """Extracts a non-negative integer (e.g. AIME math answer) from model response text."""
    if not text:
        return None

    # Priority 1: Boxed integer, e.g. \boxed{123}
    boxed_match = re.search(r"\\boxed\{\s*(-?\d+)\s*\}", text)
    if boxed_match:
        try:
            return int(boxed_match.group(1))
        except ValueError:
            pass

    # Priority 2: Explicit prefix, e.g. "The answer is 42."
    patterns = [
        r"(?:the\s+)?answer\s*(?:is|:|=)\s*[\*\_\$]*\s*(-?\d+)\b",
        r"=\s*[\*\_\$]*\s*(-?\d+)\s*[\.\$]*$",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass

    # Priority 3: Trailing integer at end of text
    trailing_match = re.search(r"(-?\d+)\s*[\.\$]*\s*$", text.strip())
    if trailing_match:
        try:
            return int(trailing_match.group(1))
        except ValueError:
            pass

    return None


def extract_linespec_answer(text: str) -> list[int] | None:
    """Extracts source line numbers or line ranges (e.g. Line 42, Lines 10-12) for COMPSEC."""
    if not text:
        return None

    # Match e.g. Line 42, lines 10-12, [10, 11]
    single_line = re.search(r"\bline\s*#?\s*(\d+)\b", text, re.IGNORECASE)
    if single_line:
        return [int(single_line.group(1))]

    range_match = re.search(r"\blines?\s*#?\s*(\d+)\s*[-–—to]+\s*(\d+)\b", text, re.IGNORECASE)
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        return list(range(start, end + 1)) if start <= end else [start]

    bracket_match = re.search(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]", text)
    if bracket_match:
        return [int(x.strip()) for x in bracket_match.group(1).split(",") if x.strip()]

    return None


def self_test_extractors() -> bool:
    """Built-in self-test verifying extractors against standard patterns."""
    assert extract_mcq_answer("Therefore, the answer is \\boxed{B}.") == "B"
    assert extract_mcq_answer("Answer: (C)") == "C"
    assert extract_mcq_answer("The correct option is D.") == "D"
    assert extract_mcq_answer("Random text without options") is None

    assert extract_integer_answer("Hence, \\boxed{123}") == 123
    assert extract_integer_answer("The answer is 42.") == 42
    assert extract_integer_answer("Final value is 0") == 0

    assert extract_linespec_answer("The vulnerability is on line 42.") == [42]
    assert extract_linespec_answer("Lines 10-12 contain the bug.") == [10, 11, 12]

    print("All extractor self-tests passed successfully.")
    return True


def main():
    ap = argparse.ArgumentParser(description="Pure text answer extractors for capability evaluation")
    ap.add_argument("--self-test-extractors", action="store_true", help="Run extractor self-tests")
    args = ap.parse_args()

    if args.self_test_extractors:
        if not self_test_extractors():
            sys.exit(1)


if __name__ == "__main__":
    main()
