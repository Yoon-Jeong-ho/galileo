#!/usr/bin/env python3
"""Append a new unchecked item to the rapid review QUEUE.md.

Design goals:
- Avoid brittle inline one-liners for cron robustness.
- Preserve existing content; only append (or create a dated section heading if missing).
- Deduplicate by URL within QUEUE.md.

Usage:
  python3 scripts/rapid_review_queue_append.py \
    --year 2026 \
    --title "Paper title" \
    --venue "arXiv" \
    --url "https://arxiv.org/abs/..." \
    --tags "tag1, tag2" \
    [--section-date 2026-02-18]

"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

QUEUE_PATH = Path("docs/paper/related_work/rapid_review/QUEUE.md")


def _today_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", required=True, type=str)
    ap.add_argument("--title", required=True, type=str)
    ap.add_argument("--venue", required=True, type=str)
    ap.add_argument("--url", required=True, type=str)
    ap.add_argument("--tags", required=True, type=str)
    ap.add_argument("--section-date", default=_today_iso(), type=str)
    args = ap.parse_args()

    if not QUEUE_PATH.exists():
        raise SystemExit(f"QUEUE.md not found at {QUEUE_PATH}")

    content = QUEUE_PATH.read_text(encoding="utf-8")

    if args.url in content:
        print(f"SKIP (already in QUEUE): {args.url}")
        return 0

    heading = f"## New candidates ({args.section_date})"
    line = (
        f"- [ ] {args.year} | {args.title} | {args.venue} | {args.url} | "
        f"tags: {args.tags}\n"
    )

    if heading not in content:
        # Append a new section at end.
        if not content.endswith("\n"):
            content += "\n"
        content += f"\n{heading}\n\n"
        content += line
    else:
        # Insert after the heading line (keep newer items at top of that section).
        lines = content.splitlines(keepends=True)
        out: list[str] = []
        inserted = False
        for i, l in enumerate(lines):
            out.append(l)
            if not inserted and l.rstrip("\n") == heading:
                # Find the first blank line after heading and insert after it.
                # If the next line isn't blank, still insert after heading.
                if i + 1 < len(lines) and lines[i + 1].strip() == "":
                    # Keep exactly one blank line, then insert.
                    # The blank line will be appended in normal flow.
                    pass
                out.append("\n" if (i + 1 >= len(lines) or lines[i + 1].strip() != "") else "")
                out.append(line)
                inserted = True
        content = "".join(out)

    QUEUE_PATH.write_text(content, encoding="utf-8")
    print(f"APPENDED: {args.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
