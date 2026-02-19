#!/usr/bin/env python3
"""Audit acronym consistency in paper-facing markdown.

Design goals:
- stdlib only (no ripgrep dependency)
- fast, single-pass scan
- actionable output (file:line + snippet)

Primary intended use (GALILEO): audit **paper-facing** files (EN draft + captions)
without drowning in internal notes/checklists.

Usage examples:

  # Paper-facing default (recommended)
  python3 scripts/audit_acronyms.py --paper-facing \
    --acronym NRC \
    --long-form "Neutral Re-asking Control" \
    --require-first-use "Neutral Re-asking Control (NRC)"

  # Broader scan (all docs/paper)
  python3 scripts/audit_acronyms.py --root docs/paper \
    --acronym NRC \
    --long-form "Neutral Re-asking Control" \
    --require-first-use "Neutral Re-asking Control (NRC)"

Exit code:
- 0: no findings
- 1: findings present
"""

from __future__ import annotations

import argparse
import fnmatch
import os
from dataclasses import dataclass


DEFAULT_PAPER_FACING = [
    "docs/paper/PAPER_DRAFT_EN.md",
    "docs/paper/FIGURE_CAPTIONS.md",
]


@dataclass(frozen=True)
class Finding:
    path: str
    line_no: int
    kind: str
    line: str


def _should_skip_dir(d: str) -> bool:
    return d in {".git", "__pycache__", "tmp", "artifacts", "figures", "paper_figures"}


def iter_text_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
        for fn in filenames:
            if fn.endswith((".md", ".tex", ".txt")):
                yield os.path.join(dirpath, fn)


def _matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in patterns)


def _load_lines(path: str) -> list[str] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.readlines()
    except UnicodeDecodeError:
        return None
    except FileNotFoundError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="docs/paper")
    ap.add_argument("--paper-facing", action="store_true", help="scan only paper-facing defaults")
    ap.add_argument("--include", action="append", default=[], help="glob to include (repeatable)")
    ap.add_argument("--exclude", action="append", default=[], help="glob to exclude (repeatable)")
    ap.add_argument("--acronym", required=True)
    ap.add_argument("--long-form", required=True)
    ap.add_argument("--require-first-use", default=None)
    ap.add_argument("--max-findings", type=int, default=200)
    args = ap.parse_args()

    acronym = args.acronym
    long_form = args.long_form
    require_first_use = args.require_first_use

    # Build candidate file list.
    if args.paper_facing:
        candidates = list(DEFAULT_PAPER_FACING)
    else:
        candidates = sorted(iter_text_files(args.root))

    # Apply include/exclude filters.
    if args.include:
        candidates = [p for p in candidates if _matches_any(p, args.include)]
    if args.exclude:
        candidates = [p for p in candidates if not _matches_any(p, args.exclude)]

    findings: list[Finding] = []

    for path in candidates:
        lines = _load_lines(path)
        if lines is None:
            continue

        blob = "".join(lines)

        # Require first-use expansion if requested.
        if require_first_use is not None:
            if (long_form in blob) and (require_first_use not in blob):
                findings.append(
                    Finding(
                        path=path,
                        line_no=1,
                        kind="missing_first_use_expansion",
                        line=f"(file contains '{long_form}' but not '{require_first_use}')",
                    )
                )

        good_expansion_prefix = f"{long_form} ({acronym}"  # allow e.g., "Neutral Re-asking Control (NRC"

        for i, line in enumerate(lines, start=1):
            # Prefer acronym after the first expansion, but allow the canonical expansion form.
            if long_form in line:
                if good_expansion_prefix not in line and (require_first_use is None or require_first_use not in line):
                    findings.append(Finding(path, i, "prefer_acronym", line.rstrip("\n")))

            # Heuristic: catch odd expansion order like "NRC (Neutral Re-asking Control)".
            if require_first_use is not None and acronym in line and long_form in line:
                if good_expansion_prefix not in line and require_first_use not in line:
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
