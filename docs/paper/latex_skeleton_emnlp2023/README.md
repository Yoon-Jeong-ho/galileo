# EMNLP 2023 template smoke test

This is a minimal smoke test that compiles our PDF figures under the official
EMNLP 2023 template (`EMNLP2023` style).

CI builds this via GitHub Actions.

## Local compile

From this directory:

```bash
pdflatex -interaction=nonstopmode main_emnlp2023.tex
bibtex main_emnlp2023
pdflatex -interaction=nonstopmode main_emnlp2023.tex
pdflatex -interaction=nonstopmode main_emnlp2023.tex
```

Notes:
- The file points to the repo-wide BibTeX database (`\\bibliography{../../../references}` → file `references.bib` at repo root), so you do not need a local copy.
- If you compile from a different working directory, ensure the relative path in `\\bibliography{...}` still resolves.

## Keeping figure snapshots in sync (SVG SSOT → PDF for LaTeX)

In this repo, paper figures are tracked as **SVG SSOT** under:
- `docs/paper/figures/*.svg`

This LaTeX skeleton includes **PDF snapshots** under:
- `docs/paper/latex_skeleton_emnlp2023/figures/*.pdf`

If you update an SVG (or regenerate figures from CSV artifacts), refresh the corresponding PDF snapshot before compiling LaTeX.

### Option A: Inkscape (recommended)

From the repo root:

```bash
# Example (adjust filenames):
inkscape docs/paper/figures/protocol_overview.svg \
  --export-type=pdf \
  --export-filename=docs/paper/latex_skeleton_emnlp2023/figures/protocol_overview.pdf
```

### Option B: rsvg-convert (if available)

```bash
rsvg-convert -f pdf -o docs/paper/latex_skeleton_emnlp2023/figures/protocol_overview.pdf \
  docs/paper/figures/protocol_overview.svg
```

(If neither tool is installed, install one locally or convert via your preferred SVG→PDF workflow. Keep the PDF filenames stable so LaTeX `\includegraphics{figures/<name>}` continues to work.)

## Anti-drift note (labels ↔ filenames)

When migrating the writing scaffold into the EMNLP template, keep LaTeX labels consistent with the repo figure filenames.
SSOT:
- `docs/paper/CLAIM_EVIDENCE_MAP.md` → “LaTeX label ↔ repo artifact mapping (anti-drift)”.

