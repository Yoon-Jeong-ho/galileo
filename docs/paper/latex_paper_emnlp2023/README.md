# GALILEO — EMNLP Main LaTeX (working)

This folder contains the **compilable** EMNLP-style main paper (`main.tex`) plus the paper figures/tables it references.

## Dependencies

You need a working LaTeX install with `latexmk` (and `pdflatex`). On Ubuntu, the simplest is `texlive-full`; a lighter setup that usually works is:

> Note: the paper will compile without Python. Python (`python3`) is only needed for the optional helper scripts that regenerate LaTeX table rows from tracked artifacts.

```bash
sudo apt-get update
sudo apt-get install -y latexmk texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra texlive-science

# Optional (only if you want to run the helper scripts):
sudo apt-get install -y python3
```

If compilation fails due to a missing `.sty`, install the corresponding TeX Live package (the error log will name it).

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

`generated/table1_rows.tex` is produced by `scripts/gen_latex_table1_from_artifacts.py`, which also tries to fill Recovery(flip) from the tracked artifact `docs/paper/artifacts/table1_recovery_from_results_paper_*.csv` when available (otherwise leaves `--`).

## Artifacts + figure regeneration (local-only)

This LaTeX paper is designed to be **auditable from tracked artifacts** in this repo.

- Tracked numeric artifacts live under: `docs/paper/artifacts/`
- Figure PDFs/PNGs referenced by `main.tex` live under: `figures/`

If you update artifacts (e.g., add a new `table1_recovery_from_results_paper_*.csv`), you can refresh the main table rows (without rerunning inference) via:

```bash
cd /path/to/repo/docs/paper/latex_paper_emnlp2023
python3 ../../../scripts/gen_latex_table1_from_artifacts.py \
  --out generated/table1_rows.tex
```

Then recompile with `latexmk` as usual.

## Optional: add uncertainty (Wilson 95% CIs) to export CSVs

Many paper exports report rates as simple `k/n` percentages (e.g., `survival_curve.csv`).
If you want lightweight confidence intervals for plotting/error bars **without rerunning any experiments**, you can use:

```bash
python3 ../../../scripts/add_wilson_ci.py \
  --in_csv  path/to/paper_exports/survival_curve.csv \
  --out_csv path/to/paper_exports/survival_curve_wilson95.csv \
  --k_col survived --n_col total \
  --rate_col survival_rate
```

For turn-of-failure histograms:

```bash
python3 ../../../scripts/add_wilson_ci.py \
  --in_csv  path/to/paper_exports/turn_of_failure.csv \
  --out_csv path/to/paper_exports/turn_of_failure_wilson95.csv \
  --k_col count --n_col total \
  --rate_col rate
```

## Page-budget markers (for scripts)

`main.tex` emits log markers like:

- `PAGE_MARK:LIMITATIONS_SECTION_START=<page>`
- `PAGE_MARK:ETHICS_SECTION_START=<page>`
- `PAGE_MARK:APPENDIX_START=<page>`

These are printed into `main.log` and can be used by helper scripts to compute page counts without manually inspecting the PDF.

## Page-budget helper script

For a one-command page count (camera-ready-ish build; main pages vs appendix), use:

```bash
cd /path/to/repo
./scripts/report_latex_page_budget.sh
```

This script regenerates `generated/table1_rows.tex` (from the paper artifacts), compiles with the `CAMERAREADY` toggle, and then reads the `PAGE_MARK:*` entries from `main.log` to report counts.

## Bibliography

The bib file is shared at the repo root:

- `../../../references.bib`
