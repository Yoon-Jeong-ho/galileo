#!/usr/bin/env python3
"""Remove a section from QUEUE.md by exact header line.

Intended for robust maintenance without brittle one-liners.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="docs/paper/related_work/rapid_review/QUEUE.md")
    ap.add_argument("--header", required=True, help="Exact header line, e.g., '## New candidates (...)'")
    args = ap.parse_args()

    queue_path = Path(args.queue)
    text = queue_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    header = args.header.rstrip("\n")

    # Find the header line.
    start = None
    for i, ln in enumerate(lines):
        if ln.rstrip("\n") == header:
            start = i
            break
    if start is None:
        print(f"header not found: {header}")
        return 0

    # Remove from the blank line before header (if present) to the line just
    # before the next '## ' header, or EOF.
    rm_start = start
    if rm_start > 0 and lines[rm_start - 1].strip() == "":
        rm_start -= 1

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break

    new_lines = lines[:rm_start] + lines[end:]
    queue_path.write_text("".join(new_lines), encoding="utf-8")
    print(f"removed section: {header}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
