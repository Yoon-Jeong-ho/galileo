#!/usr/bin/env python3
"""Fix accidental duplicated '| duplicate queue entry' segments in QUEUE.md.

Usage:
  python3 scripts/rapid_review_queue_dedupe_comment.py --url <URL>

This is intentionally conservative: it only edits lines containing the URL and
only when it sees adjacent duplicate comment tokens.
"""

import argparse
from pathlib import Path

QUEUE_PATH = Path("docs/paper/related_work/rapid_review/QUEUE.md")
TOKEN = "duplicate queue entry"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    args = ap.parse_args()

    text = QUEUE_PATH.read_text(encoding="utf-8")
    lines = text.splitlines(True)

    changed = False
    out_lines = []
    for line in lines:
        if args.url in line and TOKEN in line:
            # Collapse repeated adjacent tokens: "| duplicate queue entry | duplicate queue entry"
            while f"| {TOKEN} | {TOKEN}" in line:
                line = line.replace(f"| {TOKEN} | {TOKEN}", f"| {TOKEN}")
                changed = True
        out_lines.append(line)

    if changed:
        QUEUE_PATH.write_text("".join(out_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
