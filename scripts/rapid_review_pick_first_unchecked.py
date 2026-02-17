#!/usr/bin/env python3
"""Print the first unchecked QUEUE.md item for rapid related-work review.

Output format:
- If found: "<line_no>\t<line_text>"
- If none: "NONE"

This intentionally avoids grep/rg to reduce cron fragility.
"""

from __future__ import annotations

from pathlib import Path
import sys


def main() -> int:
    p = Path("docs/paper/related_work/rapid_review/QUEUE.md")
    if not p.exists():
        print("NONE")
        return 0

    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        s = line.lstrip()
        if s.startswith("- [ ] "):
            print(f"{i}\t{s}")
            return 0

    print("NONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
