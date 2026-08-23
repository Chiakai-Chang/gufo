#!/usr/bin/env python3
"""Fetch a bounded public-domain-style quality prompt without vendoring it."""

import argparse
from pathlib import Path
import re
import urllib.request


SOURCE_URL = "https://dgoldberg.sdsu.edu/515/harrypotter.txt"
START_MARKER = "Mr. and Mrs. Dursley"
DEFAULT_FETCH_BYTES = 16 * 1024
DEFAULT_MAX_CHARACTERS = 2200


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default=SOURCE_URL)
    parser.add_argument("--fetch-bytes", type=int, default=DEFAULT_FETCH_BYTES)
    parser.add_argument(
        "--max-characters", type=int, default=DEFAULT_MAX_CHARACTERS
    )
    args = parser.parse_args()
    if args.fetch_bytes <= 0 or args.max_characters <= 0:
        parser.error("fetch and output bounds must be positive")

    request = urllib.request.Request(
        args.url,
        headers={"Range": f"bytes=0-{args.fetch_bytes - 1}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read(args.fetch_bytes)
    text = payload.decode("utf-8", errors="replace").replace("\r\n", "\n")
    start = text.find(START_MARKER)
    if start < 0:
        raise RuntimeError(f"start marker not found in the first {args.fetch_bytes} bytes")

    excerpt = text[start : start + args.max_characters]
    sentence_end = max(
        excerpt.rfind(". "),
        excerpt.rfind("! "),
        excerpt.rfind("? "),
    )
    if sentence_end >= args.max_characters // 2:
        excerpt = excerpt[: sentence_end + 1]
    excerpt = re.sub(r"[ \t]+", " ", excerpt)
    excerpt = re.sub(r"\n{3,}", "\n\n", excerpt).strip()
    if len(excerpt) < args.max_characters // 2:
        raise RuntimeError("bounded source excerpt is unexpectedly short")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(excerpt + "\n", encoding="utf-8")
    print(
        f"wrote {args.output} characters={len(excerpt)} "
        f"source={args.url} fetched_bytes<={args.fetch_bytes}"
    )


if __name__ == "__main__":
    main()
