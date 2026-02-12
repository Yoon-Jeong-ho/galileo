# Figure captions (draft; seed1–4, Qwen2.5-7B-Instruct)

This file centralizes paper-ready captions and provenance for the **artifact-derived figures**.

Conventions:
- All figures are generated from tracked CSV artifacts under `docs/paper/artifacts/`.
- SVGs live under `docs/paper/figures/`.
- If PDFs are needed for LaTeX, generate via `scripts/convert_figures_svg_to_pdf.sh`.

Metric definitions (to keep captions consistent across drafts):
- **Survival@r**: fraction of **initially-correct** examples that remain correct for **all rounds 1..r** (cumulative; “still correct through round r”).
- **Flip**: correct→incorrect transition at some round.
- **TOF (turn-of-failure)**: the first round where a flip occurs (or “never”).

---

## Fig: Protocol overview

- File: `docs/paper/figures/protocol_overview.svg`
- LaTeX label (suggested): `fig:protocol`
- Source: generated diagram (not artifact-derived) via `scripts/make_protocol_figure_svg.py`

**Caption (draft):**
Overview of the GALILEO protocol: (1) initial evaluation on ground-truth tasks, (2) multi-round persona pressure vs Neutral Re-asking Control (drift baseline) to measure survival and turn-of-failure (TOF), and (3) recovery measured conditional on flip.

---

## Fig: Survival curves over rounds (selected personas)

- File: `docs/paper/figures/survival_curves_rounds_seed1-4_20260209.svg`
- LaTeX label (suggested): `fig:survival-curves-rounds`
- Source artifact: `docs/paper/artifacts/survival_curve_personawise_control_vs_persona_seed1-4_mean_std_20260209.csv`

**Caption (draft):**
Survival curves over interaction rounds on initially-correct examples (mean across seeds 1–4). Solid lines show persona pressure; the dashed line shows the Neutral Re-asking Control (drift baseline). Persona pressure produces heterogeneous degradation patterns, including both early-turn and late-turn failures, motivating multi-turn robustness metrics beyond initial accuracy.

---

## Fig: Persona-wise ΔSurvival@5

- File: `docs/paper/figures/survival_r5_personawise_delta_seed1-4_20260209.svg`
- LaTeX label (suggested): `fig:survival-delta-r5`
- Source artifact: `docs/paper/artifacts/survival_r5_personawise_control_vs_persona_seed1-4_mean_std_20260209.csv`

**Caption (draft):**
Persona-wise effect size at round 5: \(\Delta\)Survival@5 (persona pressure − control), mean across seeds 1–4. Negative values indicate reduced robustness under persona pressure relative to the neutral drift baseline.

---

## Fig: Persona-wise ΔFail@1 (TOF)

- File: `docs/paper/figures/tof_personawise_fail1_delta_seed1-4_20260209.svg`
- LaTeX label (suggested): `fig:tof-delta-fail1`
- Source artifact: `docs/paper/artifacts/tof_personawise_fail1_never_control_vs_persona_seed1-4_mean_std_20260209.csv`

**Caption (draft):**
Persona-wise effect size on early-turn vulnerability: \(\Delta\)Fail@1 (persona pressure − control), mean across seeds 1–4. Turn-of-failure (TOF) separates immediate flips (Fail@1) from sustained robustness (Never-fail), complementing survival curves.

---

## Fig: Persona-wise ΔRecovery@flip

- File: `docs/paper/figures/recovery_personawise_delta_seed1-4_20260209.svg`
- LaTeX label (suggested): `fig:recovery-delta`
- Source artifact: `docs/paper/artifacts/recovery_personawise_control_vs_persona_seed1-4_mean_std_20260209.csv`

**Caption (draft):**
Persona-wise effect size on recovery after flipping: \(\Delta\)Recovery@flip (persona pressure − control), mean across seeds 1–4. Recovery is measured conditional on flip, separating intervention-style “return to truth” behavior from robustness (staying correct throughout).

---

## Fig: Table W effect deltas (control vs persona)

- File: `docs/paper/figures/table_w_effect_delta_seed1-4_20260209.svg`
- LaTeX label (suggested): `fig:tablew-effect-deltas`
- Source artifact: `docs/paper/artifacts/table_w_effect_delta_seed1-4_20260209.csv`

**Caption (draft):**
Table W effect sizes: persona pressure minus Neutral Re-asking Control (drift baseline), mean across seeds 1–4. Large negative \(\Delta\)Survival@5 and positive \(\Delta\)Fail@1 indicate persona-induced failure dynamics beyond generic multi-turn drift under identical rounds/decoding/scoring.

---

## Fig: Cross-family Survival@5 (control vs Logical Trap)

- File: `docs/paper/figures/cross_family_survival_r5_control_vs_logicaltrap_seed1-2_20260212.svg`
- LaTeX label (suggested): `fig:cross-family-survival`
- Source artifacts:
  - `docs/paper/artifacts/tier1_mistral7b_seed1-2_survival_summary_20260210.csv`
  - `docs/paper/artifacts/tier1_llama3_8b_seed1-2_survival_summary_20260210.csv`
  - `docs/paper/artifacts/tier1_llama3_3b_seed1-2_survival_summary_20260212.csv`
- Generator: `scripts/make_cross_family_figure_svg.py`

**Caption (draft):**
Cross-family generalization: Survival@5 for the Neutral Re-asking Control (drift baseline) vs a strong persona (Logical Trap), averaged over seeds 1–2 for each model family. The same qualitative gap appears across families under an identical protocol.

---

## Appendix Fig (A.1): Decoding sensitivity sweep (ΔSurvival@5 and ΔFail@1)

- File: `docs/paper/figures/decoding_sweep_qwen_delta_seed1-2_20260211.svg`
- LaTeX label (suggested): `fig:decoding-sweep`
- Source artifact: `docs/paper/artifacts/decoding_sweep_qwen_temp_summary_seed1-2_20260211.csv`
- Generator: `scripts/make_decoding_sweep_figure_svg.py`

**Caption (draft):**
Appendix robustness check: decoding sensitivity for the multi-turn phase. Bars show the persona-mean effect relative to the Neutral Re-asking Control: \(\Delta\)Survival@5 and \(\Delta\)Fail@1 (persona mean − control), averaged over seeds 1–2. The persona-induced robustness gap persists across temperatures.
