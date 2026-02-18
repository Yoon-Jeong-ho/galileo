#!/usr/bin/env python3
"""Increment counters in docs/paper/related_work/rapid_review/PROGRESS.md.

Cron-robust: avoids inline one-liners and makes minimal, targeted edits.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


def bump_counter(text: str, label: str, delta: int) -> str:
    # Matches: "- <label>: <int>" with flexible whitespace.
    pattern = re.compile(rf"^(\s*-\s*{re.escape(label)}\s*:\s*)(\d+)(\s*)$", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        raise ValueError(f"Could not find counter line for: {label!r}")
    old = int(m.group(2))
    new = old + delta
    return pattern.sub(rf"\\g<1>{new}\\g<3>", text, count=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--progress",
        default="docs/paper/related_work/rapid_review/PROGRESS.md",
        help="Path to PROGRESS.md (default: %(default)s)",
    )
    ap.add_argument(
        "--papers-read-delta",
        type=int,
        default=0,
        help="Delta for 'Papers read (notes written)' counter",
    )
    ap.add_argument(
        "--top10-delta",
        type=int,
        default=0,
        help="Delta for 'Shortlisted into TOP10' counter",
    )
    args = ap.parse_args()

    path = pathlib.Path(args.progress)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return 2

    text = path.read_text(encoding="utf-8")
    if args.papers_read_delta:
        text = bump_counter(text, "Papers read (notes written)", args.papers_read_delta)
    if args.top10_delta:
        text = bump_counter(text, "Shortlisted into TOP10", args.top10_delta)

    path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
