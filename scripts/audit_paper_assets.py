#!/usr/bin/env python3
"""Audit paper-facing asset references (figures + artifacts) without LaTeX.

Why:
- Catch missing figure/artifact files early (packaging / Overleaf / anonymized bundle).
- stdlib only.

Currently checks:
- Figures: In docs/paper/PAPER_DRAFT_EN.md, find occurrences of
    \\includegraphics{figures/<name>}
  and verify that at least one of these exists locally:
    - paper_figures/pdf/<name>.pdf
    - docs/paper/figures/<name>.svg
- Artifacts: In docs/paper/PAPER_DRAFT_EN.md and docs/paper/FIGURE_CAPTIONS.md,
  find occurrences of
    docs/paper/artifacts/<name>.csv
  and verify those files exist.

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
CAPTIONS = Path("docs/paper/FIGURE_CAPTIONS.md")
SVG_DIR = Path("docs/paper/figures")
PDF_DIR = Path("paper_figures/pdf")

LATEX_MAIN = Path("docs/paper/latex_paper_emnlp2023/main.tex")
LATEX_FIG_DIR = Path("docs/paper/latex_paper_emnlp2023/figures")


INCLUDE_RE = re.compile(r"\\includegraphics\[[^\]]*\]\{figures/([^}]+)\}")
# Minimal LaTeX includegraphics matcher for our paper SSOT (no full TeX parsing).
LATEX_INCLUDE_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
ARTIFACT_RE = re.compile(r"docs/paper/artifacts/([^\s`]+?\.csv)")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> int:
    if not DRAFT.exists():
        print(f"[FAIL] missing draft: {DRAFT}")
        return 1

    draft_text = _read(DRAFT)
    caption_text = _read(CAPTIONS) if CAPTIONS.exists() else ""

    # ---- Figure assets (Markdown draft) ----
    fig_names = INCLUDE_RE.findall(draft_text)
    if not fig_names:
        print("[WARN] no \\includegraphics{figures/...} references found in markdown draft")
    else:
        missing_figs = []
        for name in sorted(set(fig_names)):
            svg = SVG_DIR / f"{name}.svg"
            pdf = PDF_DIR / f"{name}.pdf"
            if not svg.exists() and not pdf.exists():
                missing_figs.append((name, svg, pdf))

        if missing_figs:
            print(f"[FAIL] {len(missing_figs)} missing figure asset(s) referenced in markdown draft")
            for name, svg, pdf in missing_figs:
                print(f"- figures/{name}: missing both {svg} and {pdf}")
            return 1

        print(f"[OK] figures referenced in draft: {len(set(fig_names))}; all assets present")

    # ---- Figure assets (LaTeX SSOT) ----
    if LATEX_MAIN.exists():
        tex_text = _read(LATEX_MAIN)
        tex_paths = [p for p in LATEX_INCLUDE_RE.findall(tex_text) if p.startswith("figures/")]
        if not tex_paths:
            print("[WARN] no figures/... \\includegraphics references found in LaTeX main.tex")
        else:
            missing_tex = []
            for p in sorted(set(tex_paths)):
                stem = p[len("figures/") :]
                # LaTeX may resolve extensions; we check common ones we generate.
                candidates = [
                    LATEX_FIG_DIR / f"{stem}.pdf",
                    LATEX_FIG_DIR / f"{stem}.png",
                    LATEX_FIG_DIR / f"{stem}.jpg",
                    LATEX_FIG_DIR / f"{stem}.jpeg",
                    LATEX_FIG_DIR / f"{stem}.svg",
                ]
                if not any(c.exists() for c in candidates):
                    missing_tex.append((p, candidates))

            if missing_tex:
                print(f"[FAIL] {len(missing_tex)} missing figure asset(s) referenced in LaTeX main.tex")
                for p, candidates in missing_tex:
                    cand_str = ", ".join(str(c) for c in candidates)
                    print(f"- {p}: none found among [{cand_str}]")
                return 1

            print(
                f"[OK] figures referenced in LaTeX main.tex: {len(set(tex_paths))}; all assets present"
            )
    else:
        print(f"[WARN] missing LaTeX main.tex (skipping LaTeX asset audit): {LATEX_MAIN}")

    # ---- Artifact CSV assets ----
    artifact_names = ARTIFACT_RE.findall(draft_text + "\n" + caption_text)
    artifact_paths = [Path("docs/paper/artifacts") / n for n in sorted(set(artifact_names))]

    missing_artifacts = [p for p in artifact_paths if not p.exists()]
    if missing_artifacts:
        print(f"[FAIL] {len(missing_artifacts)} missing artifact CSV(s)")
        for p in missing_artifacts:
            print(f"- missing: {p}")
        return 1

    print(f"[OK] artifact CSV references: {len(artifact_paths)}; all present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
