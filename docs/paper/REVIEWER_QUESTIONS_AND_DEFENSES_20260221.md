# Reviewer questions I expect (and how we answer) — 2026-02-21

Goal: pre-empt the *highest-probability* reviewer objections with crisp, checkable answers.
This is written as if we’re responding in review/rebuttal; keep it aligned with SSOT artifacts.

---

## Q1) “Isn’t this just generic multi-turn drift / long-context degradation?”

**What they mean:** Without a matched counterfactual, persona effects are confounded with any multi-turn instability.

**Our answer:** We include a matched **Neutral Re-asking Control (NRC)** that repeats the same number of rounds and decoding settings but introduces **no new task-relevant evidence**. All persona–control comparisons are computed on the **same initially-correct subset** for that persona arm (persona-matched conditioning set), so the delta isolates pressure beyond drift.

**Where to point:**
- Protocol + NRC spec: `docs/paper/PAPER_DRAFT_EN.md` (Intro §1.1) + Fig. `fig:protocol`.
- Control-vs-persona deltas: Table~W + effect deltas figure.

**Concrete evidence (SSOT):**
- Fig: `docs/paper/figures/protocol_overview.svg`
- Table W artifact: `docs/paper/artifacts/table_w_control_vs_persona_seed1-4_mean_std_20260209.csv`
- Delta figure: `docs/paper/figures/table_w_effect_delta_seed1-4_20260209.svg`

---

## Q2) “Your Survival@r sounds like accuracy-at-r; what exactly is it?”

**What they mean:** Reviewers often confuse per-round accuracy with *survival* (cumulative correctness through all previous rounds).

**Our answer:** **Survival@r is cumulative**: the fraction of initially-correct examples that remain correct at **every** turn through round r (not just correct at round r). We define it explicitly in the Abstract and use survival curves across rounds.

**Where to point:**
- Definitions: `docs/paper/PAPER_DRAFT_EN.md` Abstract + Intro definitions.
- Survival curves: Fig.~`fig:survival-curves-rounds`.

**Concrete evidence (SSOT):**
- Fig: `docs/paper/figures/survival_curves_rounds_seed1-4_20260209.svg`
- Artifact: `docs/paper/artifacts/survival_curve_personawise_control_vs_persona_seed1-4_mean_std_20260209.csv`

---

## Q3) “TOF / Fail@1: how do you handle censoring and multi-round horizons?”

**What they mean:** They want to know if we’re mixing censored trajectories incorrectly.

**Our answer:** TOF (turn-of-failure) is the **first** incorrect turn within rounds 1..R, with right-censoring at the horizon for never-failing cases. **Fail@1** is simply \(\Pr(\mathrm{TOF}=1)\) on the initially-correct set. We report the full TOF distribution and persona-wise deltas vs NRC.

**Where to point:**
- Definitions: Abstract + protocol/method section.
- Figure: persona-wise Fail@1 deltas.

**Concrete evidence (SSOT):**
- Fig: `docs/paper/figures/tof_personawise_fail1_delta_seed1-4_20260209.svg`
- Artifact: `docs/paper/artifacts/tof_personawise_fail1_never_control_vs_persona_seed1-4_mean_std_20260209.csv`

---

## Q4) “Recovery@flip feels arbitrary / conflated with robustness. Why is it separate, and why conditional?”

**What they mean:** If recovery is just another accuracy number, it may be redundant or misleading.

**Our answer:** Staying correct (robustness) and returning to truth after a flip (recovery) are distinct behaviors. We therefore report **recovery@flip**: accuracy on a final neutral recovery turn **conditional on having flipped at least once** during rounds 1..R. This avoids conflating recovery with cases that never failed.

**Where to point:**
- Definition: Abstract (explicitly notes recovery turn excluded from survival/TOF).
- Evidence: recovery persona-wise deltas.
- Intervention: recovery prompt variants (e.g., verify_then_answer) demonstrate that recovery can change without trivially changing survival.

**Concrete evidence (SSOT):**
- Fig: `docs/paper/figures/recovery_personawise_delta_seed1-4_20260209.svg`
- Artifact: `docs/paper/artifacts/recovery_personawise_control_vs_persona_seed1-4_mean_std_20260209.csv`
- Ablation artifact: `docs/paper/artifacts/recovery_variant_verify_then_answer_vs_baseline_seed1-2_20260210.csv`

---

## Q5) “Do these effects generalize beyond one model family, or is it Qwen-specific?”

**What they mean:** They want cross-family replication and to ensure it’s not a decoding/config artifact.

**Our answer:** We run the same protocol (seeds 1–2) across multiple open-weight families and visualize survival@5 for NRC vs a strong persona (Logical Trap). We also run a decoding sensitivity check on Qwen (temperature sweep) to show the persona–control gap is qualitatively stable under sampling.

**Where to point:**
- Cross-family figure: Fig.~`fig:cross-family-survival`.
- Decoding sensitivity: Fig.~`fig:decoding-sweep` (Appendix).

**Concrete evidence (SSOT):**
- Fig: `docs/paper/figures/cross_family_survival_r5_control_vs_logicaltrap_seed1-2_20260221.svg`
- Family summary CSVs (inputs):
  - `docs/paper/artifacts/tier1_deepseek7b_seed1-2_survival_summary_20260221.csv`
  - `docs/paper/artifacts/tier1_yi6b_seed1-2_survival_summary_20260221.csv`
  - plus Mistral/Llama/Phi/Zephyr/Qwen14B snapshots listed in `docs/paper/CLAIM_EVIDENCE_MAP.md`.
- Decoding sweep fig: `docs/paper/figures/decoding_sweep_qwen_delta_seed1-2_20260211.svg`

---

## Bonus (high-risk) — “Your headline aggregate is cherry-picked / why persona-weighted?”

**Concern:** Aggregation choices can change the headline magnitude.

**Our answer:** We report both (i) a **weighted pooled-count** aggregate (used in Table W; weights effectively proportional to each persona’s evaluated set size \(|C_p|\)) and (ii) an **unweighted** mean across personas for transparency.

**Important implementation note:** In the current cross-family Tier‑1 summary pipeline (used to populate Table~1 rows for non-Qwen families), the tracked summary CSVs do **not** include per-persona denominators, so the “persona aggregate” there is implemented as an **equal-weight mean over personas of persona–NRC deltas** (matching `scripts/gen_latex_table1_from_artifacts.py`).

**Action item (writing):** Ensure each table/figure caption states which aggregation is used (pooled-count vs equal-weight-over-personas), and why.

---
