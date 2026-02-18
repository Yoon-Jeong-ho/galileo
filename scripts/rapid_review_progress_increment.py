#!/usr/bin/env python3
"""Increment counters in docs/paper/related_work/rapid_review/PROGRESS.md.

Usage:
  python3 scripts/rapid_review_progress_increment.py --papers 1
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PROGRESS_PATH = Path("docs/paper/related_work/rapid_review/PROGRESS.md")


def _inc_line(text: str, label: str, delta: int) -> str:
    # Match e.g. "- Papers read (notes written): 414"
    pattern = re.compile(rf"^(?P<prefix>\-\s+{re.escape(label)}:\s+)(?P<num>\d+)(?P<suffix>\s*)$", re.M)

    m = pattern.search(text)
    if not m:
        raise SystemExit(f"Could not find counter line for: {label}")

    num = int(m.group("num"))
    new_num = num + delta
    if new_num < 0:
        raise SystemExit(f"Refusing to make counter negative: {label} would become {new_num}")

    start, end = m.span("num")
    return text[:start] + str(new_num) + text[end:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=int, default=0, help="Increment papers-read counter by this amount")
    ap.add_argument("--top10", type=int, default=0, help="Increment TOP10-shortlist counter by this amount")
    # Back-compat: some cron attempts use --count instead of --papers.
    ap.add_argument("--count", type=int, default=0, help=argparse.SUPPRESS)
    # Cron/back-compat: accept-and-ignore common mispassed args.
    ap.add_argument("--url", default="", help=argparse.SUPPRESS)
    ap.add_argument("--note", default="", help=argparse.SUPPRESS)
    ap.add_argument("--comment", default="", help=argparse.SUPPRESS)
    args = ap.parse_args()

    # Back-compat alias.
    if args.papers == 0 and args.count:
        args.papers = args.count

    # Cron robustness: treat missing deltas as a no-op success.
    if args.papers == 0 and args.top10 == 0:
        return

    text = PROGRESS_PATH.read_text(encoding="utf-8")

    if args.papers:
        text = _inc_line(text, "Papers read (notes written)", args.papers)
    if args.top10:
        text = _inc_line(text, "Shortlisted into TOP10", args.top10)

    PROGRESS_PATH.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
