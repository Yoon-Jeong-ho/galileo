#!/usr/bin/env python3
"""Audit acronym consistency in paper-facing markdown.

Design goals:
- stdlib only (no ripgrep dependency)
- fast, single-pass scan
- actionable output (file:line + snippet)

Current checks (hardcoded for GALILEO):
- Neutral Re-asking Control should be introduced once as "Neutral Re-asking Control (NRC)".
- After introduction, prefer "NRC" (avoid repeating long form).

Usage:
  python3 scripts/audit_acronyms.py --root docs/paper --acronym NRC \
    --long-form "Neutral Re-asking Control" \
    --require-first-use "Neutral Re-asking Control (NRC)"

Exit code:
- 0: no findings
- 1: findings present
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Finding:
    path: str
    line_no: int
    kind: str
    line: str


def iter_text_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip common non-paper dirs
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", "tmp", "artifacts", "figures", "paper_figures"}]
        for fn in filenames:
            if fn.endswith((".md", ".tex", ".txt")):
                yield os.path.join(dirpath, fn)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--acronym", required=True)
    ap.add_argument("--long-form", required=True)
    ap.add_argument("--require-first-use", default=None)
    ap.add_argument("--max-findings", type=int, default=200)
    args = ap.parse_args()

    root = args.root
    acronym = args.acronym
    long_form = args.long_form
    require_first_use = args.require_first_use

    findings: list[Finding] = []

    for path in sorted(iter_text_files(root)):
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            continue

        # Require first-use expansion if requested
        if require_first_use is not None:
            if long_form in "".join(lines) and require_first_use not in "".join(lines):
                findings.append(
                    Finding(
                        path=path,
                        line_no=1,
                        kind="missing_first_use_expansion",
                        line=f"(file contains '{long_form}' but not '{require_first_use}')",
                    )
                )

        for i, line in enumerate(lines, start=1):
            if long_form in line and (require_first_use is None or require_first_use not in line):
                findings.append(Finding(path, i, "prefer_acronym", line.rstrip("\n")))
            # Heuristic: catch "NRC (Neutral Re-asking Control)" reverse expansion (prefer long-form first)
            if acronym in line and long_form in line and require_first_use is not None and require_first_use not in line:
                findings.append(Finding(path, i, "weird_expansion_order", line.rstrip("\n")))

            if len(findings) >= args.max_findings:
                break
        if len(findings) >= args.max_findings:
            break

    if findings:
        print(f"[FAIL] {len(findings)} finding(s)")
        for fnd in findings:
            rel = os.path.relpath(fnd.path, os.getcwd())
            print(f"- {fnd.kind}: {rel}:{fnd.line_no}: {fnd.line}")
        return 1

    print("[OK] no findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
