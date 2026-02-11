# EMNLP LaTeX skeleton (smoke test)

This directory is a **minimal LaTeX smoke test** to ensure our committed PDF figures
can be included cleanly in the EMNLP template.

Note: this repo environment may not have TeX installed (`pdflatex`/`latexmk`).
If TeX is missing, the skeleton still serves as a copy-pastable test for any
machine with TeX Live.

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
