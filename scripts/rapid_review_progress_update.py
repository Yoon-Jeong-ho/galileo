#!/usr/bin/env python3
"""Back-compat wrapper for older cron invocations.

Some cron runs attempted:
  python3 scripts/rapid_review_progress_update.py --delta 1

The canonical script is:
  python3 scripts/rapid_review_progress_increment.py --papers-read-delta 1

This wrapper forwards common flags to the canonical implementation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--progress", default="docs/paper/related_work/rapid_review/PROGRESS.md")
    ap.add_argument("--delta", type=int, default=0, help="Alias for papers-read delta")
    ap.add_argument("--n", type=int, default=0, help="Alias for papers-read delta")
    ap.add_argument("--papers-read-delta", type=int, default=0)
    ap.add_argument("--top10-delta", type=int, default=0)
    # Accept-and-ignore extras that may be mistakenly passed.
    ap.add_argument("--url", default="")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    papers_delta = args.papers_read_delta or args.delta or args.n

    cmd = [
        sys.executable,
        "scripts/rapid_review_progress_increment.py",
        "--progress",
        args.progress,
    ]
    if papers_delta:
        cmd += ["--papers-read-delta", str(papers_delta)]
    if args.top10_delta:
        cmd += ["--top10-delta", str(args.top10_delta)]

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
