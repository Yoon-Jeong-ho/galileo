#!/usr/bin/env python3
"""Mark the first *unchecked* queue entry matching a URL as done.

This exists because rapid_review_queue_mark_done.py may select an already
checked duplicate entry when the URL appears multiple times.

Behavior:
- Searches docs/paper/related_work/rapid_review/QUEUE.md
- Finds the first line that contains the exact URL substring AND has "- [ ]"
- Rewrites only that line:
  - changes to "- [x]"
  - appends "| <comment>" (optional)
  - appends "| note: <note>" (required)

Keeps all other lines unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

QUEUE_PATH = Path("docs/paper/related_work/rapid_review/QUEUE.md")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--note", required=True)
    ap.add_argument("--comment", default="")
    args = ap.parse_args()

    if not QUEUE_PATH.exists():
        raise SystemExit(f"Queue file not found: {QUEUE_PATH}")

    lines = QUEUE_PATH.read_text(encoding="utf-8").splitlines(True)

    target_i = None
    for i, line in enumerate(lines):
        if args.url in line and line.lstrip().startswith("- [ ]"):
            target_i = i
            break

    if target_i is None:
        # Nothing to do (already checked everywhere, or URL not present).
        return 0

    orig = lines[target_i].rstrip("\n")
    updated = orig.replace("- [ ]", "- [x]", 1)

    if args.comment:
        updated = f"{updated} | {args.comment}"

    # Add note link (ensure exactly one)
    if "| note:" not in updated:
        updated = f"{updated} | note: {args.note}"

    lines[target_i] = updated + "\n"
    QUEUE_PATH.write_text("".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
