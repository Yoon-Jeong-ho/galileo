# Paper docs (MD-only)

This folder contains the **MD-only** paper-writing workflow for the GALILEO EMNLP Main submission.

## Key files

- `PAPER_DRAFT_EN.md`: main English draft (submission-oriented wording scaffold; **SSOT for Abstract wording**)
- `ABSTRACT_EN.md`: standalone abstract draft for quick copy/paste; **keep in sync with the Abstract section in `PAPER_DRAFT_EN.md`**
- `PAPER_DRAFT_KO.md`: Korean draft/notes
- `FIGURE_CAPTIONS.md`: centralized draft captions + provenance for artifact-derived figures
- `RESULTS_INVENTORY_20260310.md`: validated March 10 result-root and export inventory for repository ops / paper-support discovery
- `REMOTE_EXPERIMENTS_RUNBOOK.md`: nlp8 experiment lane runbook (tmux/GPU/log checks + launch discipline)
- `EMNLP_MAIN_SUBMISSION_CHECKLIST.md`: submission checklist + repo/paper readiness items
- `LITERATURE_REVIEW_AND_POSITIONING_KO.md`: longer-form related-work notes + positioning
- `PAPER_RESULTS_ANALYSIS_KO.md`: quantitative results analysis notes
- `PAPER_RESULTS_QUAL_EXAMPLES_KO.md`: qualitative examples (kept separate to avoid bloating the main analysis)
- `SHORT_PAPER_README_KO.md`: readable Korean summary of whether/how this repo can support a 2-page short paper
- `SHORT_PAPER_KO.md`: figure-free Korean short-paper draft in Markdown

## 5-minute reviewer audit (reproduce key paper artifacts)

If you only have a few minutes to sanity-check the **paper-facing exports** without running new experiments, this is the shortest path.

### 1) Validate the exported results bundle

```bash
# checks runner metadata parity + basic export integrity
python3 scripts/validate_paper_exports.py --results_root results_paper --check_runner_parity
```

Expected output: a clean run and a log written under `results_paper/GLOBAL_VALIDATE.log`.

### 2) Regenerate the main paper table cells (Table 1; Survival/Fail@1/Recovery)

```bash
# Extract Table‑1-ready metrics from staged paper_exports
python3 scripts/make_table1_from_results_paper_exports.py \
  --results_paper results_paper \
  --out docs/paper/artifacts/table1_from_results_paper_exports_$(date +%Y%m%d).csv

# (Optional) regenerate LaTeX rows for Table 1
python3 scripts/gen_latex_table1_from_artifacts.py \
  --out docs/paper/latex_paper_emnlp2023/generated/table1_rows.tex
```

Notes:
- Recovery@flip is filled only when a run exported `paper_exports/recovery_accuracy.csv`.
- For missing families, rerun or re-export via `scripts/paper_export.py`.

### 3) Regenerate the paper headline table (Table W)

```bash
# produces control-vs-persona aggregates + effect deltas used in the paper
python3 scripts/make_table_w_control_vs_persona.py --control_persona_id neutral_reask_control --round 5
```

Tracked outputs live under `docs/paper/artifacts/` (CSV) and `docs/paper/figures/` (SVG) depending on the script.

### 3) Ensure figures are LaTeX-ready (optional)

```bash
# check conversion tooling + convert SVG -> PDF for LaTeX
./scripts/check_figure_tooling.sh
./scripts/convert_figures_svg_to_pdf.sh
```

### 4) Build the LaTeX paper PDF (optional)

```bash
# review-mode (line numbers)
./scripts/build_paper.sh

# camera-ready style (no line numbers; useful for page counting)
./scripts/build_paper.sh --camera-ready
```

Note: a `Makefile` exists with equivalent targets, but some minimal environments
(e.g., inside containers) may not have `make` installed; this script avoids that dependency.

### 5) Build an anonymized submission bundle (optional)

```bash
./scripts/package_anonymized_bundle.sh
# bundle staged under tmp/anonymized_bundle/
```

---

## Conventions

- Figures/tables are generated from `results/<run>/paper_exports/`.
- `docs/paper/artifacts/` contains tracked CSV artifacts used for paper claims. If a one-off/single-seed artifact becomes superseded (e.g., after adding seed2), move the older file under `docs/paper/artifacts/archive/` to reduce confusion.
- **Source-of-truth figures (vector):** `docs/paper/figures/*.svg` (generated from tracked CSV artifacts).
- **Optional submission format:** if your LaTeX pipeline prefers PDFs, generate `paper_figures/pdf/*.pdf` from the SVGs (see below). We typically **commit PDFs that are actually included in the LaTeX build** to avoid build-environment surprises.
- Prefer linking to artifact paths (CSV/SVG) from the draft so reviewers can verify claims.
- Section numbering in drafts may be renumbered/removed during LaTeX conversion.

### Metrics cheat sheet (avoid reviewer confusion)

We use **discrete-time time-to-event** language over dialogue rounds (default horizon: `R=5`) and we condition on **initial correctness**.

- **Survival@r / Survival curve**: fraction of initially-correct examples that remain correct **through** round `r` (i.e., correct at every round `1..r`).
  - This is **cumulative**; it answers “has the model ever yielded so far?”
- **Round-r accuracy**: fraction correct **at** round `r` (can be higher than Survival@r if some examples flip and later re-correct).
- **TOF (turn-of-failure)**: the **first** round `r≥1` where the *post-round model response* is incorrect; if no failure occurs within `R`, TOF is **right-censored** (*never-fail*).
- **Fail@1**: `P(TOF = 1 | initially correct)` (early-turn vulnerability).
- **Recovery@flip**: `P(correct after the Phase-3 recovery prompt | flipped at least once)`; this is evaluated **within each arm** (persona vs NRC) on that arm’s flipped subset.
- **First-passage convention** (important): once an example flips during Phase 2, it is counted as a failure for survival/TOF even if it later becomes correct again; “return-to-truth” is captured separately by Recovery@flip.

## Qualitative taxonomy labeling sheet (flip samples → manual labels)

We maintain a **reviewer-auditable** qualitative labeling workflow based on the tracked `flip_samples.csv` exports.

- Output (tracked artifact):
  - `docs/paper/artifacts/taxonomy_labeling_sheet_from_flip_samples_qwen_persona_seed1-4_20260217.csv`
- Generator script:
  - `scripts/make_taxonomy_sheet_from_flip_samples.py`
- Sampling strategy:
  - balanced across `(task_group inferred from test_name) × persona`, `--per_cell` examples each
  - deterministic given `--seed`

Usage (example):

```bash
python3 scripts/make_taxonomy_sheet_from_flip_samples.py \
  --flip_csvs results/<RUN1>/paper_exports/flip_samples.csv,results/<RUN2>/paper_exports/flip_samples.csv \
  --out_csv docs/paper/artifacts/taxonomy_labeling_sheet_from_flip_samples_<MODEL>_seed1-2_<YYYYMMDD>.csv \
  --per_cell 10 \
  --seed 42
```

## Figure inventory (current)

All figures below are **generated from tracked artifacts** and stored under `docs/paper/figures/`.

- Survival curves over rounds (selected personas; includes dashed control baseline):
  - `docs/paper/figures/survival_curves_rounds_seed1-4_20260209.svg`
  - Source artifact: `docs/paper/artifacts/survival_curve_personawise_control_vs_persona_seed1-4_mean_std_20260209.csv`
- Persona-wise effect size at round 5 (ΔSurvival@5):
  - `docs/paper/figures/survival_r5_personawise_delta_seed1-4_20260209.svg`
  - Source artifact: `docs/paper/artifacts/survival_r5_personawise_control_vs_persona_seed1-4_mean_std_20260209.csv`
- Persona-wise early-turn vulnerability (ΔFail@1):
  - `docs/paper/figures/tof_personawise_fail1_delta_seed1-4_20260209.svg`
  - Source artifact: `docs/paper/artifacts/tof_personawise_fail1_never_control_vs_persona_seed1-4_mean_std_20260209.csv`
- Persona-wise recovery after flipping (ΔRecovery@flip):
  - `docs/paper/figures/recovery_personawise_delta_seed1-4_20260209.svg`
  - Source artifact: `docs/paper/artifacts/recovery_personawise_control_vs_persona_seed1-4_mean_std_20260209.csv`
- Table W: control vs persona effect deltas (Δ metrics):
  - `docs/paper/figures/table_w_effect_delta_seed1-4_20260209.svg`
  - Source artifact: `docs/paper/artifacts/table_w_effect_delta_seed1-4_20260209.csv`

## Figure conversion (SVG → PDF)

**Recommendation (default): use PDF in LaTeX builds.** Overleaf/LaTeX is most reliable with PDF/PNG figures. We keep SVG as the source of truth and generate PDFs as needed.

### If PDF conversion is blocked (no root / no `rsvg-convert`)

If your build environment cannot install system packages (no sudo), you have three options:

1) **No-sudo AppImage conversion (recommended in this repo):** download a pinned Inkscape AppImage via `bash scripts/get_inkscape_appimage.sh`, then run `./scripts/convert_figures_svg_to_pdf.sh`.

2) **Convert SVG→PDF elsewhere:** run conversion on any machine that has `rsvg-convert` (Ubuntu `librsvg2-bin`) or `inkscape`, then copy the resulting `paper_figures/pdf/*.pdf` into your LaTeX build.

3) **SVG-in-LaTeX fallback (brittle):** use the LaTeX `svg` package and compile with `--shell-escape` so LaTeX can call an external converter.
   - This can be convenient locally, but it is more fragile across build systems and may be disallowed by some conference build pipelines.
   - If using Overleaf, you may need to enable shell-escape (project settings) and ensure the backend supports conversion.

We prefer option (1) for EMNLP submission packaging because it avoids relying on shell-escape.

After generating PDFs, if your LaTeX project expects `figures/*.pdf` under a TeX directory (e.g., Overleaf), sync our generated PDFs into the repo’s LaTeX skeletons via:

```bash
bash scripts/sync_pdf_figures_to_latex_skeleton.sh
```

- Script: `scripts/convert_figures_svg_to_pdf.sh`
- Preflight check: `scripts/check_figure_tooling.sh`
- Requires **one** of:
  - `rsvg-convert` (recommended; `librsvg2-bin` on Ubuntu), or
  - `inkscape`

Usage:

```bash
# default: docs/paper/figures/*.svg -> paper_figures/pdf/*.pdf
./scripts/convert_figures_svg_to_pdf.sh
```

## Anonymized bundle (pre-submission / external share)

To avoid accidentally leaking hostnames/absolute paths in an anonymized submission/artifact bundle:

- Script: `scripts/package_anonymized_bundle.sh`
- Stages a minimal bundle under `tmp/anonymized_bundle/` and **fails fast** if infra-identifying strings are found.

```bash
./scripts/package_anonymized_bundle.sh
# then zip/tar tmp/anonymized_bundle/
```

## Sanity checks (fast)

- **Citation key check (avoid LaTeX build failures):**

```bash
# robust: catches \cite{...}, \citet{...}, \citep{...}, and optional pre/post notes
bash scripts/check_citations_vs_bib.sh
```

This scans the main Markdown drafts plus all `*.tex` under `docs/paper/` (excluding the upstream EMNLP template) and checks that every citation key exists in `references.bib`.

- **Legacy/quick check (limited):**

```bash
python3 tools/check_bibkeys.py
```

Note: this only matches plain `\cite{...}` (it will miss `\citet`/`\citep`), so prefer `scripts/check_citations_vs_bib.sh` for paper readiness.

## Make targets (optional)

If you prefer `make` (note: some minimal environments may not have `make` installed), you can run:

```bash
make figures-check
make figures-pdf
make anonymized-bundle
```

If `make` is unavailable, run the scripts directly:

```bash
./scripts/check_figure_tooling.sh
./scripts/convert_figures_svg_to_pdf.sh
./scripts/package_anonymized_bundle.sh
```

For what to sanitize/exclude, see `docs/paper/ANONYMIZATION_NOTES.md`.

## Automation notes (OpenClaw heartbeat)

If you use OpenClaw heartbeats to produce periodic writing updates:

- Keep the workspace `HEARTBEAT.md` non-empty (not just headers/blank lines), otherwise heartbeats may be skipped.
- Avoid running large debug `exec` commands during heartbeats: failures (e.g., `pipefail`/SIGPIPE from `| head`) can generate noisy “Exec failed” DM messages.

### Naming conventions to keep paper ↔ exports aligned

- **Drift baseline name (paper):** *Neutral Re-asking Control*.
- **Recommended export identifier:** use a stable tag like `neutral_reask_control` (either as `persona=neutral_reask_control` or `condition=neutral_reask_control`) so plotting scripts can auto-style it (e.g., dashed line) and include it by default.
- **Recommended run metadata:** export a small `paper_exports/metadata.json` containing decoding params, seeds, git commit hash, and the control/persona identifier mapping so “identical settings” claims are auditable.
- **Avoid confusion with Simple Denial:** Simple Denial is an *adversarial persona*; the Neutral Re-asking Control is *non-adversarial* and is only meant to capture generic multi-turn drift.
