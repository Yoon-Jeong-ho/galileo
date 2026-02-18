#!/usr/bin/env python3
"""Remove queue entries that contain any of the given URLs.

This is meant for robust cleanup when bad candidates are appended.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="docs/paper/related_work/rapid_review/QUEUE.md")
    ap.add_argument("--url", action="append", required=True, help="URL substring to match; may be repeated")
    args = ap.parse_args()

    queue_path = Path(args.queue)
    lines = queue_path.read_text(encoding="utf-8").splitlines(keepends=True)

    urls = args.url

    def is_bad(ln: str) -> bool:
        return any(u in ln for u in urls)

    new_lines = [ln for ln in lines if not is_bad(ln)]

    # Also drop any immediately-preceding header if it becomes followed by a blank
    # line and then another header (i.e., empty section). Do a simple pass.
    cleaned: list[str] = []
    i = 0
    while i < len(new_lines):
        if new_lines[i].startswith("## "):
            # look ahead to find the next non-empty line
            j = i + 1
            while j < len(new_lines) and new_lines[j].strip() == "":
                j += 1
            if j < len(new_lines) and new_lines[j].startswith("## "):
                # empty section; skip this header line
                i += 1
                continue
        cleaned.append(new_lines[i])
        i += 1

    queue_path.write_text("".join(cleaned), encoding="utf-8")
    removed = len(lines) - len(new_lines)
    print(f"removed_lines={removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
