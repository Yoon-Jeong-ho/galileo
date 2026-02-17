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
- The file points to the repo-wide BibTeX database (`../../../references.bib`) so you do not need a local copy.
- If you compile from a different working directory, ensure the relative path in `\\bibliography{...}` still resolves.

## Anti-drift note (labels ↔ filenames)

When migrating the writing scaffold into the EMNLP template, keep LaTeX labels consistent with the repo figure filenames.
SSOT:
- `docs/paper/CLAIM_EVIDENCE_MAP.md` → “LaTeX label ↔ repo artifact mapping (anti-drift)”.

