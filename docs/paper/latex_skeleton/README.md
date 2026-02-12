# EMNLP LaTeX skeleton (smoke test)

This directory is a **minimal LaTeX smoke test** to ensure our committed PDF figures
can be included cleanly in the EMNLP template.

Note: this repo environment may not have TeX installed (`pdflatex`/`latexmk`).
If TeX is missing, the skeleton still serves as a copy-pastable test for any
machine with TeX Live.

## Anti-drift note (labels ↔ filenames)

Figure labels in the draft/LaTeX are easy to drift away from the repo figure filenames.
Use the SSOT mapping in:
- `docs/paper/CLAIM_EVIDENCE_MAP.md` → “LaTeX label ↔ repo artifact mapping (anti-drift)” section.

## Quick test (on a machine with TeX)

```bash
cd docs/paper/latex_skeleton
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

If `latexmk` is available:

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```
