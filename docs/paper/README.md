# Paper docs (MD-only)

This folder contains the **MD-only** paper-writing workflow for the GALILEO EMNLP Main submission.

## Key files

- `PAPER_DRAFT_EN.md`: main English draft (submission-oriented wording scaffold)
- `PAPER_DRAFT_KO.md`: Korean draft/notes
- `EMNLP_MAIN_SUBMISSION_CHECKLIST.md`: submission checklist + repo/paper readiness items
- `LITERATURE_REVIEW_AND_POSITIONING_KO.md`: longer-form related-work notes + positioning
- `PAPER_RESULTS_ANALYSIS_KO.md`: quantitative results analysis notes
- `PAPER_RESULTS_QUAL_EXAMPLES_KO.md`: qualitative examples (kept separate to avoid bloating the main analysis)

## Conventions

- Figures/tables are generated from `results/<run>/paper_exports/` and stored under `paper_figures/`.
- Prefer linking to artifact paths (CSV/SVG) from the draft so reviewers can verify claims.
- Section numbering in drafts may be renumbered/removed during LaTeX conversion.

## Automation notes (OpenClaw heartbeat)

If you use OpenClaw heartbeats to produce periodic writing updates:

- Keep the workspace `HEARTBEAT.md` non-empty (not just headers/blank lines), otherwise heartbeats may be skipped.
- Avoid running large debug `exec` commands during heartbeats: failures (e.g., `pipefail`/SIGPIPE from `| head`) can generate noisy “Exec failed” DM messages.

### Naming conventions to keep paper ↔ exports aligned

- **Drift baseline name (paper):** *Neutral Re-asking Control*.
- **Recommended export identifier:** use a stable tag like `neutral_reask_control` (either as `persona=neutral_reask_control` or `condition=neutral_reask_control`) so plotting scripts can auto-style it (e.g., dashed line) and include it by default.
- **Recommended run metadata:** export a small `paper_exports/metadata.json` containing decoding params, seeds, git commit hash, and the control/persona identifier mapping so “identical settings” claims are auditable.
- **Avoid confusion with Simple Denial:** Simple Denial is an *adversarial persona*; the Neutral Re-asking Control is *non-adversarial* and is only meant to capture generic multi-turn drift.
