#!/usr/bin/env python3
"""Audit paper-facing asset references (figures + artifacts) without LaTeX.

Why:
- Catch missing figure/artifact files early (packaging / Overleaf / anonymized bundle).
- stdlib only.

Currently checks:
- In docs/paper/PAPER_DRAFT_EN.md, find occurrences of
    \\includegraphics{figures/<name>}
  and verify that at least one of these exists locally:
    - paper_figures/pdf/<name>.pdf
    - docs/paper/figures/<name>.svg

Notes:
- We intentionally do not parse full LaTeX; we use a simple regex that matches
  the subset we write in the draft.

Usage:
  python3 scripts/audit_paper_assets.py

Exit code:
- 0: OK
- 1: missing assets
"""

from __future__ import annotations

import re
from pathlib import Path


DRAFT = Path("docs/paper/PAPER_DRAFT_EN.md")
SVG_DIR = Path("docs/paper/figures")
PDF_DIR = Path("paper_figures/pdf")


INCLUDE_RE = re.compile(r"\\includegraphics\[[^\]]*\]\{figures/([^}]+)\}")


def main() -> int:
    if not DRAFT.exists():
        print(f"[FAIL] missing draft: {DRAFT}")
        return 1

    text = DRAFT.read_text(encoding="utf-8")
    names = INCLUDE_RE.findall(text)

    if not names:
        print("[WARN] no \\includegraphics{figures/...} references found")
        return 0

    missing = []
    for name in sorted(set(names)):
        svg = SVG_DIR / f"{name}.svg"
        pdf = PDF_DIR / f"{name}.pdf"
        if not svg.exists() and not pdf.exists():
            missing.append((name, svg, pdf))

    if missing:
        print(f"[FAIL] {len(missing)} missing figure asset(s)")
        for name, svg, pdf in missing:
            print(f"- figures/{name}: missing both {svg} and {pdf}")
        return 1

    print(f"[OK] figures referenced in draft: {len(set(names))}; all assets present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
