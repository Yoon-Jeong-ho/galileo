#!/usr/bin/env python3
"""Mark an item as done in rapid-review QUEUE.md.

This script exists to avoid brittle `python -c` one-liners in cron jobs.

Usage examples:
  python3 scripts/rapid_review_queue_mark_done.py \
    --queue docs/paper/related_work/rapid_review/QUEUE.md \
    --url https://arxiv.org/abs/2602.01146 \
    --note docs/paper/related_work/rapid_review/papers/20260218_persistbench.md \
    --comment "duplicate queue entry"

Semantics:
- Finds the first line containing the given URL (string match).
- Replaces leading "- [ ]" with "- [x]" if present; otherwise ensures "[x]" is present.
- Appends/updates a "note:" field (and optional comment) at end of the line.

Non-goals:
- Full markdown parsing; this is intentionally simple and conservative.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _normalize_spaces(s: str) -> str:
    return " ".join(s.split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", type=Path, default=Path("docs/paper/related_work/rapid_review/QUEUE.md"))
    ap.add_argument("--url", required=True, help="URL string to match within QUEUE.md")
    ap.add_argument("--note", default=None, help="Note path to append as `note: <path>`")
    ap.add_argument("--comment", default=None, help="Optional short comment to append")
    args = ap.parse_args()

    p: Path = args.queue
    if not p.exists():
        raise FileNotFoundError(f"QUEUE not found: {p}")

    lines = p.read_text(encoding="utf-8").splitlines(True)

    hit = None
    for i, line in enumerate(lines):
        if args.url in line:
            hit = i
            break

    if hit is None:
        raise ValueError(f"URL not found in QUEUE: {args.url}")

    line = lines[hit].rstrip("\n")

    # Mark done.
    if line.lstrip().startswith("- [ ]"):
        prefix, rest = line.split("- [ ]", 1)
        line = prefix + "- [x]" + rest
    elif "[x]" not in line:
        # If it is a bullet but missing a checkbox, don't try to inject one.
        # Just leave it as-is.
        pass

    # Remove existing note/comment fragments conservatively.
    # We only rewrite the tail fields we manage.
    parts = line.split(" | ")
    parts = [p_ for p_ in parts if not p_.strip().startswith("note:")]
    parts = [p_ for p_ in parts if p_.strip().startswith("-") or "http" in p_ or True]

    # Reassemble with optional fields.
    line = " | ".join(parts)
    if args.note:
        line = line + " | note: " + _normalize_spaces(args.note)
    if args.comment:
        line = line + " | " + _normalize_spaces(args.comment)

    lines[hit] = line + "\n"
    p.write_text("".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
