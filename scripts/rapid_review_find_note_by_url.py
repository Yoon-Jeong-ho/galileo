#!/usr/bin/env python3
"""Find an existing rapid-review note by matching a URL string.

Usage:
  python3 scripts/rapid_review_find_note_by_url.py --url <URL>

Output:
- If found: prints the note path (relative to repo root), preferring the newest file.
- If not found: prints NOTHING and exits 0.

We keep this intentionally simple (string match) to support cron de-dup.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    args = ap.parse_args()

    url = args.url.strip()
    if not url:
        return 0

    root = Path("docs/paper/related_work/rapid_review/papers")
    if not root.exists():
        return 0

    matches: list[Path] = []
    for p in sorted(root.glob("*.md")):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if url in txt:
            matches.append(p)

    if not matches:
        return 0

    # Prefer newest by name (YYYYMMDD_* convention).
    best = sorted(matches, key=lambda x: x.name)[-1]
    print(str(best))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
