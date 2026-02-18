#!/usr/bin/env python3
"""Canonicalize and de-duplicate rapid-review QUEUE.md by URL.

Motivation
----------
The rapid-review cron has occasionally re-added already-reviewed papers with URL
variants (e.g., arXiv /abs vs /pdf, trailing slashes) and/or duplicate entries.
This script:
1) Canonicalizes common URL variants (currently: arXiv) to https://arxiv.org/abs/<id>(vN)
2) Removes subsequent duplicate entries (same canonical URL) from QUEUE.md

It is intentionally conservative:
- Only touches lines that look like queue items: start with '- [ ]' or '- [x]'
  and contain a '| <URL> |' field.
- Preserves non-item lines and comments verbatim.

Usage
-----
  python3 scripts/rapid_review_queue_canonicalize_and_dedupe.py
  python3 scripts/rapid_review_queue_canonicalize_and_dedupe.py --inplace

By default it prints the would-be updated content to stdout.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Keep this in sync with scripts/rapid_review_find_note_by_url.py
_ARXIV_ID_RE = re.compile(r"(?P<id>\d{4}\.\d{4,5})(?P<v>v\d+)?", re.IGNORECASE)


def _canonicalize_url(url: str) -> str:
    url = url.strip()
    if not url:
        return url

    # Normalize simple trailing slash.
    url = url.rstrip("/")

    # Canonicalize arXiv.
    if "arxiv.org" in url:
        m = _ARXIV_ID_RE.search(url)
        if m:
            arxiv_id = m.group("id")
            ver = m.group("v") or ""
            return f"https://arxiv.org/abs/{arxiv_id}{ver}"

    return url


_ITEM_RE = re.compile(r"^(?P<prefix>\s*-\s*\[(?:x| )\]\s*)(?P<rest>.*)$")


def _split_fields(rest: str) -> list[str] | None:
    # Queue format uses pipes; require at least 5 fields (year/title/venue/url/tags...)
    parts = [p.strip() for p in rest.split("|")]
    if len(parts) < 5:
        return None
    return parts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--queue",
        default="docs/paper/related_work/rapid_review/QUEUE.md",
        help="Path to QUEUE.md",
    )
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="Write changes back to QUEUE.md (default: print to stdout)",
    )
    args = ap.parse_args()

    q = Path(args.queue)
    text = q.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=False)

    seen: set[str] = set()
    out_lines: list[str] = []
    removed = 0
    canon_changed = 0

    for line in lines:
        m = _ITEM_RE.match(line)
        if not m:
            out_lines.append(line)
            continue

        rest = m.group("rest")
        parts = _split_fields(rest)
        if not parts:
            out_lines.append(line)
            continue

        # Find the first URL-looking field (we expect field[3] by convention).
        # Use the first part that starts with http.
        url_idx = None
        for i, p in enumerate(parts):
            if p.startswith("http://") or p.startswith("https://"):
                url_idx = i
                break
        if url_idx is None:
            out_lines.append(line)
            continue

        old_url = parts[url_idx]
        new_url = _canonicalize_url(old_url)
        if new_url != old_url:
            parts[url_idx] = new_url
            canon_changed += 1

        canon = new_url
        if canon in seen:
            removed += 1
            continue
        seen.add(canon)

        rebuilt = m.group("prefix") + " | ".join(parts)
        out_lines.append(rebuilt)

    out = "\n".join(out_lines) + ("\n" if text.endswith("\n") else "")

    if args.inplace:
        q.write_text(out, encoding="utf-8")
        print(f"[OK] wrote {q} (removed {removed} duplicates; canonicalized {canon_changed} URLs)")
    else:
        print(out, end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
