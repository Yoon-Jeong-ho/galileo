# GALILEO — EMNLP Main LaTeX (working)

This folder contains the **compilable** EMNLP-style main paper (`main.tex`) plus the paper figures/tables it references.

## Quick build

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

This defaults to **review mode** (line numbers), which is convenient for drafting.

## Camera-ready (no line numbers)

`main.tex` supports a simple toggle via `\def\CAMERAREADY{1}`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -pdflatex='pdflatex %O "\\def\\CAMERAREADY{1}\\input{%S}"' \
  main.tex
```

## Where the table rows come from

`main.tex` prefers generated rows when present:

- `generated/table1_rows.tex` (preferred)
- `static/table1_rows.tex` (fallback)

So local compiles should succeed even if you haven't regenerated exports.

## Page-budget markers (for scripts)

`main.tex` emits log markers like:

- `PAGE_MARK:LIMITATIONS_SECTION_START=<page>`
- `PAGE_MARK:ETHICS_SECTION_START=<page>`
- `PAGE_MARK:APPENDIX_START=<page>`

These are printed into `main.log` and can be used by helper scripts to compute page counts without manually inspecting the PDF.

## Bibliography

The bib file is shared at the repo root:

- `../../../references.bib`
