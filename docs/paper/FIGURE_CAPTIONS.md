# Figure captions (draft; seed1–4, Qwen2.5-7B-Instruct)

This file centralizes paper-ready captions and provenance for the **artifact-derived figures**.

Conventions:
- All figures are generated from tracked CSV artifacts under `docs/paper/artifacts/`.
- SVGs live under `docs/paper/figures/`.
- If PDFs are needed for LaTeX, generate via `scripts/convert_figures_svg_to_pdf.sh`.
- Unless otherwise stated, error bars / uncertainty annotations reflect **variation across random seeds** (reported as mean ± std).
- Unless otherwise stated, all multi-turn robustness metrics (Survival/TOF/Fail@1/Recovery@flip) are computed on the **initially-correct subset** (conditioning on correctness at round 0). As a result:
  - The effective sample size can differ across personas/seeds/tasks; captions should avoid implying a fixed global \(n\).
  - When comparing persona pressure to the **Neutral Re-asking Control (NRC)**, we use a **matched conditioning set**: the control arm is evaluated on the same initially-correct subset as the persona arm. This prevents conditioning-set drift from being mistaken as a treatment effect.

Neutral Re-asking Control (NRC; a.k.a. “drift baseline”): re-asks the same task over multiple rounds with *neutral* prompts (no persona pressure) to estimate generic multi-turn drift under the same rounds/decoding/scoring.

Caption boilerplate (optional, to keep wording consistent):
- “Computed on the initially-correct subset; control is evaluated on the same persona-matched subset. Error bars are mean ± std over seeds.”

Caption style notes (paper-ready):
- First mention expands abbreviations (TOF, Fail@1, etc.).
- Captions explicitly state the comparison direction for \(\Delta\) metrics (persona − control).

Metric definitions (to keep captions consistent across drafts):
- **Table W** (canonical LaTeX label: `tab:tablew`): pooled control vs persona summary + effect deltas (see artifacts `table_w_*`). **Important:** Table W is a *collapsed* view that pools across persona arms *after* computing persona-vs-control on the same persona-matched initially-correct subset (\(C_p\)) within each arm, so “control” values can differ from the control shown in persona-wise figures/tables.

  In the tracked artifacts we report multiple aggregation conventions (these can yield slightly different headline numbers; captions/text should state which is being cited):
  - **delta-first macro (paper default):** compute the persona--NRC delta within each (persona, task) cell first, then take an equal-weight mean across cells (optionally then mean±std across seeds). This avoids implicit dataset-size weighting and keeps drift-correction aligned to each persona’s matched control before pooling.
  - **persona_weighted (pooled-counts):** *pool counts across personas first* (equivalently, weight each persona by its evaluation-set size). Concretely, for a rate metric expressed as a numerator/denominator (e.g., Survival@5 = survived/total), we compute
    \[
    \text{persona\_weighted} = \frac{\sum_p \mathrm{num}_p}{\sum_p \mathrm{den}_p},
    \]
    where \(p\) ranges over persona arms and \((\mathrm{num}_p,\mathrm{den}_p)\) are computed on that persona’s matched initially-correct subset \(C_p\) (and the NRC is evaluated on the same \(C_p\) before pooling). For TOF-style histograms, we pool by summing per-round counts across personas.
  - **persona_unweighted:** simple mean of persona-wise rates (each persona counts equally).

  Default for this captions file: follow the paper default unless a specific artifact/figure is explicitly defined as pooled-counts (persona_weighted).
- **Survival@r**: fraction of **initially-correct** examples that remain correct for **all rounds 1..r** (cumulative; “still correct through round r”).
- **Flip**: a correct→incorrect transition at some round **within the multi-turn phase** (rounds 1..R). The final neutral **recovery turn** is *not* counted when defining flips/TOF.
- **TOF (turn-of-failure)**: the first round (within 1..R) where a flip occurs. If an example remains correct through the horizon (rounds 1..R), we record **TOF = “never”** and treat it as **right-censored** at R (time-to-event framing).
- **Fail@1**: probability of flipping at the **first** pressure round (i.e., TOF=1).
- **Effect deltas** (caption shorthand): by default, \(\Delta\text{Metric}\) means **persona pressure minus NRC** under an identical protocol (same rounds/decoding/scoring). Deltas are computed on the **initially-correct subset**, so that survival/TOF/recovery reflect robustness dynamics rather than initial capability.
  - **When an artifact defines persona aggregates via pooled counts** (e.g., Table W; decoding sweep summaries), we compute the aggregated persona rate and aggregated control rate *within each seed* by pooling numerators/denominators across personas (as specified above), then take their difference; error bars report mean ± std across seeds.
  - **When an artifact only exposes per-persona rates/deltas without denominators** (e.g., Tier-1 cross-family `tier1_*_survival_summary_*.csv`), any “persona aggregate” is computed as an **equal-weight mean over personas** of the persona–NRC deltas (then reconstructed as NRC + mean(delta)), matching `scripts/gen_latex_table1_from_artifacts.py`.
  - \(\Delta\text{Survival@5} < 0\): reduced robustness under persona pressure.
  - \(\Delta\text{Fail@1} > 0\): increased immediate vulnerability.
  - \(\Delta\text{Recovery@flip} < 0\): worse return-to-truth conditional on flip.

---

## Fig: Protocol overview

- File: `docs/paper/figures/protocol_overview.svg`
- LaTeX label (suggested): `fig:protocol`
- Source: generated diagram (not artifact-derived) via `scripts/make_protocol_figure_svg.py`

**Caption (draft):**
Overview of the GALILEO protocol: (1) initial evaluation on ground-truth tasks, (2) multi-round persona pressure vs NRC (drift baseline) to measure survival and turn-of-failure (TOF), and (3) recovery measured conditional on flip. Robustness metrics are computed on the initially-correct subset (conditioning on round-0 correctness); the control arm is evaluated on the same persona-matched subset to isolate multi-turn drift from conditioning-set differences.

---

## Fig: Survival curves over rounds (selected personas)

- File: `docs/paper/figures/survival_curves_rounds_seed1-4_20260209.svg`
- LaTeX label (suggested): `fig:survival-curves-rounds`
- Source artifact: `docs/paper/artifacts/survival_curve_personawise_control_vs_persona_seed1-4_mean_std_20260209.csv`

**Caption (draft):**
Survival curves over interaction rounds on the initially-correct subset (mean ± std across seeds 1–4). Solid lines: persona pressure. Dashed lines: NRC (drift baseline), evaluated on the same persona-matched initially-correct subset. Persona pressure induces heterogeneous failure dynamics (early-turn vs late-turn flips), motivating multi-turn robustness metrics beyond initial accuracy.

---

## Fig: Persona-wise ΔSurvival@5

- File: `docs/paper/figures/survival_r5_personawise_delta_seed1-4_20260209.svg`
- LaTeX label (suggested): `fig:survival-delta-r5`
- Source artifact: `docs/paper/artifacts/survival_r5_personawise_control_vs_persona_seed1-4_mean_std_20260209.csv`

**Caption (draft):**
Persona-wise effect size at round 5: \(\Delta\)Survival@5 (persona pressure − control), computed on the initially-correct subset; the control arm is evaluated on the same persona-matched subset. Error bars are mean ± std across seeds 1–4. Negative values indicate reduced robustness under persona pressure relative to the neutral drift baseline.

---

## Fig: Persona-wise ΔFail@1 (TOF)

- File: `docs/paper/figures/tof_personawise_fail1_delta_seed1-4_20260209.svg`
- LaTeX label (suggested): `fig:tof-delta-fail1`
- Source artifact: `docs/paper/artifacts/tof_personawise_fail1_never_control_vs_persona_seed1-4_mean_std_20260209.csv`

**Caption (draft):**
Persona-wise effect size on early-turn vulnerability: \(\Delta\)Fail@1 (persona pressure − control), computed on the initially-correct subset; the control arm is evaluated on the same persona-matched subset. Error bars are mean ± std across seeds 1–4. Fail@1 summarizes the turn-of-failure (TOF) distribution at round 1 (immediate flip); paired with the “never-fail” mass, it distinguishes early-turn brittleness from sustained robustness, complementing survival curves.

---

## Fig: Persona-wise ΔRecovery@flip

- File: `docs/paper/figures/recovery_personawise_delta_seed1-4_20260209.svg`
- LaTeX label (suggested): `fig:recovery-delta`
- Source artifact: `docs/paper/artifacts/recovery_personawise_control_vs_persona_seed1-4_mean_std_20260209.csv`

**Caption (draft):**
Persona-wise effect size on recovery after flipping: \(\Delta\)Recovery@flip (persona pressure − control), computed on the initially-correct subset; the control arm is evaluated on the same persona-matched subset. Error bars are mean ± std across seeds 1–4. Recovery is measured conditional on flip, separating intervention-style “return to truth” behavior from robustness (staying correct throughout).

---

## Fig: Table W effect deltas (control vs persona)

- File: `docs/paper/figures/table_w_effect_delta_seed1-4_20260209.svg`
- LaTeX label (suggested): `fig:tablew-effect-deltas`
- Source artifact: `docs/paper/artifacts/table_w_effect_delta_seed1-4_20260209.csv`

**Caption (draft):**
Table W effect sizes (persona pressure minus NRC, drift baseline). Unless otherwise stated, we use the paper-default **delta-first macro** aggregation: compute persona--NRC deltas within each (persona, task) cell on its matched initially-correct set \(C_p\), then average deltas across cells (mean ± std across seeds 1–4). This keeps drift-correction aligned to each persona’s paired control before pooling and avoids implicit dataset-size weighting.

(Alternative view available in artifacts: **persona_weighted** pooled-counts aggregation, weighting personas by \(|C_p|\). Table W also includes a persona-unweighted aggregate for transparency.)

---

## Fig: Cross-family Survival@5 (control vs Logical Trap)

- File: `docs/paper/figures/cross_family_survival_r5_control_vs_logicaltrap_seed1-2_20260221.svg`
- LaTeX label (suggested): `fig:cross-family-survival`
- Source artifacts:
  - `docs/paper/artifacts/tier1_mistral7b_seed1-2_survival_summary_20260210.csv`
  - `docs/paper/artifacts/tier1_mistralnemo_seed1-2_survival_summary_20260217.csv`
  - `docs/paper/artifacts/tier1_llama3_8b_seed1-2_survival_summary_20260210.csv`
  - `docs/paper/artifacts/tier1_llama3_3b_seed1-2_survival_summary_20260212.csv`
  - `docs/paper/artifacts/tier1_phi3mini_seed1-2_survival_summary_20260217.csv`
  - `docs/paper/artifacts/tier1_phi35mini_seed1-2_survival_summary_20260219.csv`
  - `docs/paper/artifacts/tier1_zephyr7b_seed1-2_survival_summary_20260218.csv`
  - `docs/paper/artifacts/tier1_qwen2p5_14b_seed1-2_survival_summary_20260219.csv`
  - `docs/paper/artifacts/tier1_deepseek7b_seed1-2_survival_summary_20260221.csv`
  - `docs/paper/artifacts/tier1_yi6b_seed1-2_survival_summary_20260221.csv`
- Generator: `scripts/make_cross_family_figure_svg.py`

**Caption (draft):**
Cross-family generalization: Survival@5 for the NRC (drift baseline) vs a strong persona (Logical Trap), computed on the initially-correct subset; the control arm is evaluated on the same persona-matched subset. Error bars are mean ± std over seeds 1–2 for each model family. The same qualitative gap appears across families under an identical protocol. For some families we cap `max_model_len` for KV-cache feasibility on the available hardware (e.g., Mistral-Nemo at 32k); rounds, personas, decoding, and scoring are otherwise identical.

---

## Appendix Fig (A.1): Decoding sensitivity sweep (ΔSurvival@5 and ΔFail@1)

- File: `docs/paper/figures/decoding_sweep_qwen_delta_seed1-2_20260211.svg`
- LaTeX label (suggested): `fig:decoding-sweep`
- Source artifact: `docs/paper/artifacts/decoding_sweep_qwen_temp_summary_seed1-2_20260211.csv`
- Generator: `scripts/make_decoding_sweep_figure_svg.py`

**Caption (draft):**
Appendix robustness check: decoding sensitivity for the multi-turn phase. Bars show the persona-mean effect relative to the NRC: \(\Delta\)Survival@5 and \(\Delta\)Fail@1 (persona − control), computed on the initially-correct subset with the control arm evaluated on the same persona-matched subset. Unless otherwise noted, the “persona mean” is the persona-weighted aggregate (pooling with weights proportional to persona set size). Error bars are mean ± std over seeds 1–2. The persona-induced robustness gap persists across temperatures.
