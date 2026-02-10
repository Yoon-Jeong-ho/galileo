# Paper docs (MD-only)

This folder contains the **MD-only** paper-writing workflow for the GALILEO EMNLP Main submission.

## Key files

- `PAPER_DRAFT_EN.md`: main English draft (submission-oriented wording scaffold)
- `PAPER_DRAFT_KO.md`: Korean draft/notes
- `FIGURE_CAPTIONS.md`: centralized draft captions + provenance for artifact-derived figures
- `REMOTE_EXPERIMENTS_RUNBOOK.md`: nlp16 experiment lane runbook (tmux/GPU/log checks + launch discipline)
- `EMNLP_MAIN_SUBMISSION_CHECKLIST.md`: submission checklist + repo/paper readiness items
- `LITERATURE_REVIEW_AND_POSITIONING_KO.md`: longer-form related-work notes + positioning
- `PAPER_RESULTS_ANALYSIS_KO.md`: quantitative results analysis notes
- `PAPER_RESULTS_QUAL_EXAMPLES_KO.md`: qualitative examples (kept separate to avoid bloating the main analysis)

## Conventions

- Figures/tables are generated from `results/<run>/paper_exports/`.
- **Source-of-truth figures (vector):** `docs/paper/figures/*.svg` (generated from tracked CSV artifacts).
- **Optional submission format:** if your LaTeX pipeline prefers PDFs, generate `paper_figures/pdf/*.pdf` from the SVGs (see below).
- Prefer linking to artifact paths (CSV/SVG) from the draft so reviewers can verify claims.
- Section numbering in drafts may be renumbered/removed during LaTeX conversion.

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

If your build environment cannot install system packages (no sudo), you have two options:

1) **Convert SVG→PDF elsewhere** (recommended): run conversion on any machine that has `rsvg-convert` (Ubuntu `librsvg2-bin`) or `inkscape`, then copy the resulting `paper_figures/pdf/*.pdf` into your LaTeX build.

2) **SVG-in-LaTeX fallback (brittle):** use the LaTeX `svg` package and compile with `--shell-escape` so LaTeX can call an external converter.
   - This can be convenient locally, but it is more fragile across build systems and may be disallowed by some conference build pipelines.
   - If using Overleaf, you may need to enable shell-escape (project settings) and ensure the backend supports conversion.

We prefer option (1) for EMNLP submission packaging because it avoids relying on shell-escape.

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
