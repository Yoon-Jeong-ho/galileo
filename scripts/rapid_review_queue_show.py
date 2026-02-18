#!/usr/bin/env python3
"""Show a specific QUEUE.md item (by line number) for debugging/cron logs.

This replaces brittle inline Python or missing-helper failures.

Usage:
  python3 scripts/rapid_review_queue_show.py --line 737
  python3 scripts/rapid_review_queue_show.py --line 737 --context 2

Behavior:
- Prints the requested line plus a small context window.
- Exits non-zero if the line is out of range.

We keep it intentionally simple (no markdown parsing).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--queue",
        type=Path,
        default=Path("docs/paper/related_work/rapid_review/QUEUE.md"),
        help="Path to QUEUE.md",
    )
    ap.add_argument("--line", type=int, required=True, help="1-indexed line number to show")
    ap.add_argument("--context", type=int, default=1, help="Number of lines of context above/below")
    args = ap.parse_args()

    if args.line <= 0:
        raise ValueError("--line must be >= 1")
    if args.context < 0:
        raise ValueError("--context must be >= 0")

    p: Path = args.queue
    txt = p.read_text(encoding="utf-8").splitlines()
    n = len(txt)

    idx = args.line - 1
    if idx < 0 or idx >= n:
        print(f"[ERROR] line out of range: {args.line} (file has {n} lines): {p}")
        return 2

    lo = max(0, idx - args.context)
    hi = min(n, idx + args.context + 1)

    print(f"QUEUE: {p}")
    for i in range(lo, hi):
        prefix = ">" if i == idx else " "
        print(f"{prefix}{i+1:5d}: {txt[i]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
