#!/usr/bin/env python3
"""Find an existing rapid-review note by matching a URL.

Usage:
  python3 scripts/rapid_review_find_note_by_url.py --url <URL>

Output:
- If found: prints the note path (relative to repo root), preferring the newest file.
- If not found: prints NOTHING and exits 0.

Notes
-----
We normalize common URL variants (especially arXiv abs/pdf/html differences) so
cron de-dup doesn't regress into repeated "duplicate" processing.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urlparse, urlunparse


_ARXIV_ID_RE = re.compile(r"(?P<id>\d{4}\.\d{4,5})(?P<v>v\d+)?", re.IGNORECASE)


def _strip_query_fragment(url: str) -> str:
    try:
        u = urlparse(url)
    except Exception:
        return url
    u2 = u._replace(query="", fragment="")
    return urlunparse(u2)


def _canonical_arxiv_abs(url: str) -> str | None:
    """Return canonical arXiv abs URL if url looks like arXiv; else None."""
    u = urlparse(url)
    host = (u.netloc or "").lower()
    path = (u.path or "")

    if "arxiv.org" not in host:
        return None

    m = _ARXIV_ID_RE.search(path)
    if not m:
        return None

    arxiv_id = m.group("id")
    ver = m.group("v") or ""
    return f"https://arxiv.org/abs/{arxiv_id}{ver}"


def normalize_url(url: str) -> tuple[str, ...]:
    """Return a small set of normalized URL variants to match in text."""
    url = url.strip()
    if not url:
        return tuple()

    url = _strip_query_fragment(url)

    # Always include the stripped original.
    variants = {url.rstrip("/")}

    # Canonicalize arXiv links to abs form (handles /pdf/<id>.pdf, /html/<id>, etc.).
    can = _canonical_arxiv_abs(url)
    if can:
        variants.add(can)

    return tuple(sorted(v for v in variants if v))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    args = ap.parse_args()

    needles = normalize_url(args.url)
    if not needles:
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
        if any(n in txt for n in needles):
            matches.append(p)

    if not matches:
        return 0

    # Prefer newest by name (YYYYMMDD_* convention).
    best = sorted(matches, key=lambda x: x.name)[-1]
    print(str(best))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
