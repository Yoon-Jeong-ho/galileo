#!/usr/bin/env python3
"""Mark the first *unchecked* (- [ ]) QUEUE.md entry for a given URL as done.

Rationale: the queue may contain the same URL multiple times; the standard
mark_done helper may touch an already-completed entry depending on ordering.

Usage:
  python3 scripts/rapid_review_queue_mark_done_first_unchecked.py --url <URL> --note <path> [--comment "..."]

Behavior:
- Finds the first line containing the URL and the token "- [ ]".
- Replaces it with "- [x]".
- Appends a "note: <path>" field if missing.
- Appends the optional comment (verbatim) if provided and not already present.
"""

import argparse
from pathlib import Path

QUEUE_PATH = Path("docs/paper/related_work/rapid_review/QUEUE.md")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--note", required=True)
    ap.add_argument("--comment", default="")
    args = ap.parse_args()

    lines = QUEUE_PATH.read_text(encoding="utf-8").splitlines(True)

    changed = False
    for i, line in enumerate(lines):
        if args.url in line and "- [ ]" in line:
            newline = line.replace("- [ ]", "- [x]", 1)

            if "| note:" not in newline:
                # Preserve trailing newline
                end = "\n" if newline.endswith("\n") else ""
                core = newline[:-1] if end else newline
                core = core.rstrip()
                core = core + f" | note: {args.note}"
                newline = core + end

            if args.comment:
                if args.comment not in newline:
                    end = "\n" if newline.endswith("\n") else ""
                    core = newline[:-1] if end else newline
                    core = core.rstrip()
                    core = core + f" | {args.comment}"
                    newline = core + end

            lines[i] = newline
            changed = True
            break

    if not changed:
        raise SystemExit(f"No unchecked entry found for URL: {args.url}")

    QUEUE_PATH.write_text("".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
