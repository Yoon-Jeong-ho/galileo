#!/usr/bin/env python3
r"""Audit LaTeX citation keys used in paper drafts against references.bib.

Purpose:
- Catch missing BibTeX entries early (before LaTeX compile time).
- Keep drafts consistent as we replace author-year placeholders with \cite{...}.

Default scope:
- docs/paper/PAPER_DRAFT_EN.md
- docs/paper/PAPER_DRAFT_KO.md (if present)
- references.bib

Usage:
  python3 scripts/audit_citations.py

Optional:
  python3 scripts/audit_citations.py --paths docs/paper/latex_paper_emnlp2023/main.tex

Note:
- By default, this script scans only the markdown drafts above.
  You can additionally scan LaTeX SSOT files (e.g., docs/paper/latex_paper_emnlp2023/main.tex)
  by passing --paths.

Exit codes:
- 0: all cited keys exist in references.bib
- 2: missing keys detected
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable, Set

ROOT = Path(__file__).resolve().parents[1]


def bib_keys(bib_path: Path) -> Set[str]:
    text = bib_path.read_text(encoding="utf-8")
    # Simple BibTeX key capture: @type{key,
    return set(re.findall(r"@\w+\{([^,\s]+)", text))


def cited_keys(md_path: Path) -> Set[str]:
    text = md_path.read_text(encoding="utf-8")
    keys: Set[str] = set()
    # supports \cite{...}, \citet{...}, \citep{...}
    for m in re.finditer(r"\\cite\w*\{([^}]+)\}", text):
        for k in m.group(1).split(","):
            k = k.strip()
            if k:
                keys.add(k)
    return keys


def main() -> int:
    bib = ROOT / "references.bib"
    if not bib.exists():
        print(f"[ERROR] missing {bib}")
        return 2

    bibk = bib_keys(bib)

    # Default paths: markdown drafts (paper-facing).
    default_paths = [
        ROOT / "docs" / "paper" / "PAPER_DRAFT_EN.md",
        ROOT / "docs" / "paper" / "PAPER_DRAFT_KO.md",
    ]
    default_paths = [p for p in default_paths if p.exists()]

    # Optional override/extension.
    paths = default_paths
    if "--paths" in sys.argv:
        # Lightweight parse (avoid argparse churn for this tiny script).
        i = sys.argv.index("--paths")
        extra = [Path(x) for x in sys.argv[i + 1 :] if not x.startswith("-")]
        extra = [x if x.is_absolute() else (ROOT / x) for x in extra]
        # Merge while preserving order and uniqueness.
        merged = [p for p in paths if p.exists()]
        for p in extra:
            if p.exists() and p not in merged:
                merged.append(p)
        paths = merged

    if not paths:
        print("[ERROR] no input paths found")
        return 2

    ok = True
    for d in paths:
        rel = d.relative_to(ROOT) if d.is_relative_to(ROOT) else d
        ck = cited_keys(d)
        missing = sorted(ck - bibk)
        print(f"== {rel}")
        print(f"  cited keys: {len(ck)}")
        if missing:
            ok = False
            print(f"  [MISSING] {len(missing)}")
            for k in missing:
                print(f"    - {k}")
        else:
            print("  [OK] all cited keys present in references.bib")

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
