# Figure captions (draft; seed1–4, Qwen2.5-7B-Instruct)

This file centralizes paper-ready captions and provenance for the **artifact-derived figures**.

Conventions:
- All figures are generated from tracked CSV artifacts under `docs/paper/artifacts/`.
- SVGs live under `docs/paper/figures/`.
- If PDFs are needed for LaTeX, generate via `scripts/convert_figures_svg_to_pdf.sh`.

---

## Fig: Protocol overview

- File: `docs/paper/figures/protocol_overview.svg`
- Source: generated diagram (not artifact-derived) via `scripts/make_protocol_figure_svg.py`

**Caption (draft):**
Overview of the GALILEO evaluation protocol. Phase 1 identifies the initially-correct subset (C) on ground-truth tasks. Phase 2 applies multi-round persona pressure or a Neutral Re-asking Control (drift baseline) to quantify robustness over turns via survival and turn-of-failure (TOF). Phase 3 measures recovery conditional on flip, capturing a distinct “return to truth” axis.

---

## Fig: Survival curves over rounds (selected personas)

- File: `docs/paper/figures/survival_curves_rounds_seed1-4_20260209.svg`
- Source artifact: `docs/paper/artifacts/survival_curve_personawise_control_vs_persona_seed1-4_mean_std_20260209.csv`

**Caption (draft):**
Survival curves over interaction rounds on initially-correct examples (mean across seeds 1–4). Solid lines show persona pressure; the dashed line shows the Neutral Re-asking Control (drift baseline). Persona pressure produces heterogeneous degradation patterns, including both early-turn and late-turn failures, motivating multi-turn robustness metrics beyond initial accuracy.

---

## Fig: Persona-wise ΔSurvival@5

- File: `docs/paper/figures/survival_r5_personawise_delta_seed1-4_20260209.svg`
- Source artifact: `docs/paper/artifacts/survival_r5_personawise_control_vs_persona_seed1-4_mean_std_20260209.csv`

**Caption (draft):**
Persona-wise effect size at round 5: \(\Delta\)Survival@5 (persona pressure − control), mean across seeds 1–4. Negative values indicate reduced robustness under persona pressure relative to the neutral drift baseline.

---

## Fig: Persona-wise ΔFail@1 (TOF)

- File: `docs/paper/figures/tof_personawise_fail1_delta_seed1-4_20260209.svg`
- Source artifact: `docs/paper/artifacts/tof_personawise_fail1_never_control_vs_persona_seed1-4_mean_std_20260209.csv`

**Caption (draft):**
Persona-wise effect size on early-turn vulnerability: \(\Delta\)Fail@1 (persona pressure − control), mean across seeds 1–4. Turn-of-failure (TOF) separates immediate flips (Fail@1) from sustained robustness (Never-fail), complementing survival curves.

---

## Fig: Persona-wise ΔRecovery@flip

- File: `docs/paper/figures/recovery_personawise_delta_seed1-4_20260209.svg`
- Source artifact: `docs/paper/artifacts/recovery_personawise_control_vs_persona_seed1-4_mean_std_20260209.csv`

**Caption (draft):**
Persona-wise effect size on recovery after flipping: \(\Delta\)Recovery@flip (persona pressure − control), mean across seeds 1–4. Recovery is measured conditional on flip, separating intervention-style “return to truth” behavior from robustness (staying correct throughout).

---

## Fig: Table W effect deltas (control vs persona)

- File: `docs/paper/figures/table_w_effect_delta_seed1-4_20260209.svg`
- Source artifact: `docs/paper/artifacts/table_w_effect_delta_seed1-4_20260209.csv`

**Caption (draft):**
Table W effect sizes: persona pressure minus Neutral Re-asking Control (drift baseline), mean across seeds 1–4. Large negative \(\Delta\)Survival@5 and positive \(\Delta\)Fail@1 indicate persona-induced failure dynamics beyond generic multi-turn drift under identical rounds/decoding/scoring.
