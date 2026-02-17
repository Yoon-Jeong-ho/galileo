#!/usr/bin/env python3
"""Commit untracked rapid-review notes to avoid broken references on fresh clones.

Why this exists:
- The rapid-review cron sometimes leaves behind a note file under
  docs/paper/related_work/rapid_review/papers/ as untracked (e.g., when a run
  tries to stay scoped to exactly one paper).
- That creates repo hygiene issues because QUEUE.md may link to the file.

Behavior:
- If there are untracked files under docs/paper/related_work/rapid_review/papers/
  matching today's YYYYMMDD_*.md (or any *.md), stage+commit them.
- Exits 0 if nothing to do.

This script is intentionally conservative: it only touches that notes directory.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def sh(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def main() -> int:
    root = Path("docs/paper/related_work/rapid_review/papers")
    if not root.exists():
        return 0

    # Get untracked files (??) from porcelain status.
    out = sh(["git", "status", "--porcelain"])
    paths: list[str] = []
    for line in out.splitlines():
        if not line.startswith("?? "):
            continue
        p = line[3:]
        if p.startswith(str(root) + "/") and p.endswith(".md"):
            paths.append(p)

    if not paths:
        return 0

    # Stage and commit.
    subprocess.check_call(["git", "add", "--"] + paths)
    msg = "rapid-review: cleanup untracked rapid-review notes"
    subprocess.check_call(["git", "commit", "-m", msg])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
