#!/usr/bin/env python3
"""Mark a specific line number as done in rapid-review QUEUE.md.

Motivation: URLs can appear multiple times; the default mark_done script
marks the first occurrence. This script targets an exact line number.

Usage:
  python3 scripts/rapid_review_queue_mark_done_by_line.py --line 803 \
    --note docs/paper/related_work/rapid_review/papers/20260217_x.md \
    --comment "duplicate queue entry"

Semantics:
- Operates on QUEUE.md by default.
- Marks checkbox "- [ ]" -> "- [x]" on that line if present.
- Removes any existing trailing fields that start with "note:".
- Appends "note: <path>" and optional comment as " | <comment>".

This is intentionally conservative (no full markdown parsing).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _normalize_spaces(s: str) -> str:
    return " ".join(s.split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", type=Path, default=Path("docs/paper/related_work/rapid_review/QUEUE.md"))
    ap.add_argument("--line", type=int, required=True, help="1-indexed line number in QUEUE.md")
    ap.add_argument("--note", default=None, help="Note path to append as `note: <path>`")
    ap.add_argument("--comment", default=None, help="Optional short comment to append")
    args = ap.parse_args()

    p: Path = args.queue
    if not p.exists():
        raise FileNotFoundError(f"QUEUE not found: {p}")

    lines = p.read_text(encoding="utf-8").splitlines(True)
    idx = args.line - 1
    if idx < 0 or idx >= len(lines):
        raise IndexError(f"Line out of range: {args.line} (file has {len(lines)} lines)")

    raw = lines[idx].rstrip("\n")

    # Only operate on list items.
    if "- [" not in raw:
        raise ValueError(f"Target line does not look like a checkbox item: {args.line}: {raw}")

    line = raw
    if line.lstrip().startswith("- [ ]"):
        prefix, rest = line.split("- [ ]", 1)
        line = prefix + "- [x]" + rest

    # Remove existing managed tails.
    parts = line.split(" | ")
    parts = [p_ for p_ in parts if not p_.strip().startswith("note:")]
    # Also avoid duplicating the exact same comment repeatedly.
    if args.comment:
        norm_comment = _normalize_spaces(args.comment)
        parts = [p_ for p_ in parts if _normalize_spaces(p_) != norm_comment]

    line = " | ".join(parts)
    if args.note:
        line = line + " | note: " + _normalize_spaces(args.note)
    if args.comment:
        line = line + " | " + _normalize_spaces(args.comment)

    lines[idx] = line + "\n"
    p.write_text("".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
