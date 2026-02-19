# Claim → Evidence Map (GALILEO EMNLP Main)

Purpose: make the paper *reviewer-auditable* by mapping each core claim to:
1) where it is stated in the draft,
2) what figure/table substantiates it (artifact path), and
3) what script regenerates that artifact.

This file is intentionally short and pragmatic (SSOT for “what proves what”).

---

## Terminology guardrails (avoid reviewer confusion)

- **Survival(p, r)**: probability the model remains correct **for every round 1..r** (cumulative), *not* “accuracy at round r only.”
  - In tracked artifacts, we store both the **full curve** (`survival_curve_personawise_*`) and the **round-5 summary** (`survival_r5_personawise_*`). Keep proof pointers consistent about which one is being cited.
- **Flip**: a trajectory event **correct → incorrect** between rounds.
- **TOF (turn-of-failure)**: the **first round index** where a flip occurs; “Fail@1” is the probability that TOF = 1.
- **Recovery**: correctness **after** a flip (distinct from survival/TOF); reported separately.

These definitions should be used consistently in the draft, captions, and artifacts.

- **Important (control comparability / reporting modes):** we use two matched-set reporting modes, and the draft must say which one is being used.
  - **persona-matched** (within-persona attribution; *used by our main tracked artifacts, including Table~W*): for each persona arm we filter to that persona’s *initially-correct* subset `C_p` and run both (i) persona pressure and (ii) the Neutral Re-asking Control on that same subset. This makes persona-vs-control comparisons apples-to-apples *within* a persona, but **control numbers can differ across personas** because the underlying `C_p` differs. Table~W then reports persona aggregates by pooling (weighted or unweighted) across these persona-wise matched comparisons.
  - **shared-\(C\)** (clean cross-persona comparisons): for each seed we define a single initially-correct subset `C` under a persona-free neutral prompt and evaluate *every* persona arm and the Neutral Re-asking Control on exactly the same `C`. This keeps the control baseline fixed across personas and avoids mixing different conditioning sets when directly ranking personas.

- **Important (Table W aggregation; avoid “why doesn’t this match the persona-wise deltas?” confusion):** Table W intentionally reports **two** persona aggregates:
  - **persona\_weighted:** pool counts across personas first (equivalently: weight each persona by its evaluation-set size; implemented by summing `survived/total` at round `R` and summing TOF counts across personas).
  - **persona\_unweighted:** simple mean of persona-wise rates.

  **Caption template (recommended):**
  > *Table W: Control vs persona pressure on initially-correct examples.* We compare the Neutral Re-asking Control (drift baseline) to persona pressure using (i) Survival@R, (ii) Fail@1, and (iii) Never-fail, averaged over seeds (mean±std). For persona pressure we report both a **weighted** aggregate (pooling across personas with weights proportional to each persona’s evaluation-set size) and an **unweighted** aggregate (simple mean across personas). Weighted is the headline measure; unweighted is included for transparency.

---

## LaTeX label ↔ repo artifact mapping (anti-drift)

When converting the draft to LaTeX, keep these **label → file** mappings stable so proof pointers remain correct.

- `fig:protocol` → `docs/paper/figures/protocol_overview.svg`
- `fig:survival-curves-rounds` → `docs/paper/figures/survival_curves_rounds_seed1-4_20260209.svg`
- `fig:tof-delta-fail1` → `docs/paper/figures/tof_personawise_fail1_delta_seed1-4_20260209.svg`
- `fig:recovery-delta` → `docs/paper/figures/recovery_personawise_delta_seed1-4_20260209.svg`
- `fig:tablew-effect-deltas` → `docs/paper/figures/table_w_effect_delta_seed1-4_20260209.svg`
- `tab:tablew` → (Table W; artifacts) `docs/paper/artifacts/table_w_control_vs_persona_seed1-4_mean_std_20260209.csv` and `docs/paper/artifacts/table_w_effect_delta_seed1-4_20260209.csv`
- `fig:cross-family-survival` → `docs/paper/figures/cross_family_survival_r5_control_vs_logicaltrap_seed1-2_20260219.svg`
- `fig:decoding-sweep` → `docs/paper/figures/decoding_sweep_qwen_delta_seed1-2_20260211.svg`

(If the figure filenames change due to regeneration, update **both** this map and the LaTeX labels in the draft in the same commit.)

---

## Abstract/Intro (reviewer-auditable checklist)

These are the claims most likely to be read *without* looking at appendices. Each should have an obvious proof pointer.

### Abstract: sentence-level claim map (keep this in sync with `PAPER_DRAFT_EN.md`)

**A0 (setup; why this matters).**
- Draft text (Abstract sentence 1–2): “LLMs can abandon previously correct answers under social/rhetorical pressure … yet most benchmarks report only single-turn accuracy, obscuring interaction dynamics.”
- What must be true / measurable in our paper: multi-turn correctness can decay over rounds under pressure.
- Proof pointers:
  - Survival dynamics: Fig.~`fig:survival-curves-rounds` → `docs/paper/figures/survival_curves_rounds_seed1-4_20260209.svg`
  - Early-turn vulnerability: Fig.~`fig:tof-delta-fail1` → `docs/paper/figures/tof_personawise_fail1_delta_seed1-4_20260209.svg`

**A1 (what we introduce).**
- Draft text (Abstract sentence 3): “We introduce GALILEO … and compare against a matched Neutral Re-asking Control.”
- Proof pointers:
  - Protocol diagram: Fig.~`fig:protocol` → `docs/paper/figures/protocol_overview.svg`
  - Control-vs-persona summary: Table~`tab:tablew` + Fig.~`fig:tablew-effect-deltas`
    - Artifacts: `docs/paper/artifacts/table_w_control_vs_persona_seed1-4_mean_std_20260209.csv`, `docs/paper/artifacts/table_w_effect_delta_seed1-4_20260209.csv`

**A2 (what we measure; definitions).**
- Draft text (Abstract sentence 4): “We report survival curves, TOF/Fail@1, and recovery@flip … interpret flips as time-to-event with right-censoring.”
- Proof pointers (definitions + empirical instantiations):
  - Survival: Fig.~`fig:survival-curves-rounds`; artifact `docs/paper/artifacts/survival_curve_personawise_control_vs_persona_seed1-4_mean_std_20260209.csv`
  - TOF / Fail@1: Fig.~`fig:tof-delta-fail1`; artifact `docs/paper/artifacts/tof_personawise_fail1_never_control_vs_persona_seed1-4_mean_std_20260209.csv`
  - Recovery@flip: Fig.~`fig:recovery-delta`; artifact `docs/paper/artifacts/recovery_personawise_control_vs_persona_seed1-4_mean_std_20260209.csv`

**A3 (headline finding; mechanism vs drift).**
- Draft text (Abstract sentence 5): “Across multi-seed experiments … persona pressure reduces survival relative to Neutral Re-asking Control and can induce early-turn vulnerability.”
- Proof pointers:
  - Persona-vs-control deltas: Table~`tab:tablew` + Fig.~`fig:tablew-effect-deltas`
  - Dynamics evidence: Fig.~`fig:survival-curves-rounds`; Fig.~`fig:tof-delta-fail1`

**A4 (headline finding; robustness vs recovery are distinct).**
- Draft text (Abstract sentence 6): “Recovery@flip varies by task and persona … staying correct vs returning to truth are distinct behaviors.”
- Proof pointers:
  - Recovery deltas: Fig.~`fig:recovery-delta`
  - (If we make task-stratified claims) ensure corresponding task-wise artifacts/figures are cited in the Results section; do not rely on the abstract alone.

### Intro: proof-pointer hooks (keep these aligned)

These are the exact “proof pointer” hooks we want a reviewer to notice in the Introduction; if you edit them in the draft, update the mapping here in the same commit.

- **Protocol + drift baseline:** Introduction §1.1 should point to Fig.~`fig:protocol` plus Table~`tab:tablew` and Fig.~`fig:tablew-effect-deltas`. It should also include the *within-persona initially-correct subset* comparability caveat (control values can differ across personas).
- **Dynamics + recovery:** Introduction §1.1 should explicitly name survival/TOF/recovery and point to Figs.~`fig:survival-curves-rounds`, `fig:tof-delta-fail1`, `fig:recovery-delta` plus Table~`tab:tablew`.
- **Evaluation gap bullets:** Introduction §1.2 bullets should each have a single obvious proof pointer (TOF → `fig:tof-delta-fail1`, survival → `fig:survival-curves-rounds`, recovery → `fig:recovery-delta`).
- **Intro skim hook:** Introduction §1.1 should contain an “Evidence at a glance” bullet list pointing to (i) protocol+control, (ii) survival+TOF, and (iii) recovery (Figs.~`fig:protocol`, `fig:survival-curves-rounds`, `fig:tof-delta-fail1`, `fig:recovery-delta`; Table~`tab:tablew`).

### Abstract/Intro: claim-level checklist (high-level, for quick audit)

1) **Multi-turn persona pressure degrades robustness over rounds** (not captured by single-turn accuracy).
   - Evidence: `docs/paper/figures/survival_curves_rounds_seed1-4_20260209.svg`
   - Artifact: `docs/paper/artifacts/survival_curve_personawise_control_vs_persona_seed1-4_mean_std_20260209.csv`
   - Regenerate: `python3 scripts/make_paper_figures_from_artifacts.py` (reads tracked CSVs under `docs/paper/artifacts/` and overwrites SVGs under `docs/paper/figures/`; safe/idempotent).
2) **Failures happen early (TOF / Fail@1 changes) and the effect is persona-dependent.**
   - Evidence: `docs/paper/figures/tof_personawise_fail1_delta_seed1-4_20260209.svg`
   - Artifact: `docs/paper/artifacts/tof_personawise_fail1_never_control_vs_persona_seed1-4_mean_std_20260209.csv`
   - Regenerate: `python3 scripts/make_paper_figures_from_artifacts.py`
3) **Neutral Re-asking Control separates generic drift from persona-induced failures.**
   - Evidence: Table W + deltas figure `docs/paper/figures/table_w_effect_delta_seed1-4_20260209.svg`
   - Artifact: `docs/paper/artifacts/table_w_control_vs_persona_seed1-4_mean_std_20260209.csv` and `docs/paper/artifacts/table_w_effect_delta_seed1-4_20260209.csv`
   - Regenerate:
     1) `python3 scripts/make_table_w_control_vs_persona.py` (writes/updates the Table-W CSV artifacts under `docs/paper/artifacts/`)
     2) `python3 scripts/make_paper_figures_from_artifacts.py` (renders/overwrites the SVG figure under `docs/paper/figures/`; safe/idempotent)
4) **Recovery after flipping is distinct and measurable (not implied by survival/TOF).**
   - Evidence: `docs/paper/figures/recovery_personawise_delta_seed1-4_20260209.svg`
   - Artifact: `docs/paper/artifacts/recovery_personawise_control_vs_persona_seed1-4_mean_std_20260209.csv`
   - Regenerate: `python3 scripts/make_paper_figures_from_artifacts.py`
5) **Cross-family replication under the same protocol (at least seeds 1–2).**
   - Evidence: `docs/paper/figures/cross_family_survival_r5_control_vs_logicaltrap_seed1-2_20260219.svg`
   - Artifact inputs (CSV → figure):
     - `docs/paper/artifacts/tier1_mistral7b_seed1-2_survival_summary_20260210.csv`
     - `docs/paper/artifacts/tier1_mistralnemo_seed1-2_survival_summary_20260217.csv`
     - `docs/paper/artifacts/tier1_llama3_8b_seed1-2_survival_summary_20260210.csv`
     - `docs/paper/artifacts/tier1_llama3_3b_seed1-2_survival_summary_20260212.csv`
     - `docs/paper/artifacts/tier1_phi3mini_seed1-2_survival_summary_20260217.csv`
     - `docs/paper/artifacts/tier1_zephyr7b_seed1-2_survival_summary_20260218.csv`
   - Regenerate: `python3 scripts/make_cross_family_figure_svg.py`
6) **Decoding sensitivity check: persona-vs-control gaps are qualitatively stable under sampling.**
   - Evidence: `docs/paper/figures/decoding_sweep_qwen_delta_seed1-2_20260211.svg`
   - Artifact: `docs/paper/artifacts/decoding_sweep_qwen_temp_summary_seed1-2_20260211.csv`
   - Regenerate: `python3 scripts/make_decoding_sweep_figure_svg.py`

---

## C1 (Dynamics): Robustness under pressure is a trajectory (survival + TOF), not a single number

**Where stated**
- `docs/paper/PAPER_DRAFT_EN.md` → §1.4 Core claims (C1)

**Primary evidence (paper-facing)**
- Survival curves across rounds (persona vs control):
  - Artifact (CSV): `docs/paper/artifacts/survival_curve_personawise_control_vs_persona_seed1-4_mean_std_20260209.csv`
  - Figure (SVG): `docs/paper/figures/survival_curves_rounds_seed1-4_20260209.svg`
  - Regeneration: `python3 scripts/make_paper_figures_from_artifacts.py` (reads `docs/paper/artifacts/*`)
- Early-turn failure / TOF deltas (persona-wise Fail@1):
  - Artifact (CSV): `docs/paper/artifacts/tof_personawise_fail1_never_control_vs_persona_seed1-4_mean_std_20260209.csv`
  - Figure (SVG): `docs/paper/figures/tof_personawise_fail1_delta_seed1-4_20260209.svg`
  - Regeneration: `python3 scripts/make_paper_figures_from_artifacts.py`

**Sanity / audit hooks**
- Paper exports per run (required for auditable aggregation): `paper_exports/` + `metadata.json` + `runner_metadata.json`
- Validator: `python3 scripts/validate_paper_exports.py`

---

## C2 (Mechanism vs drift): Persona pressure induces failures beyond generic multi-turn drift

**Where stated**
- `docs/paper/PAPER_DRAFT_EN.md` → §1.4 Core claims (C2)

**Primary evidence**
- Table W (control vs persona) + effect deltas (ΔSurvival@5, ΔFail@1, etc.):
  - Artifact (CSV): `docs/paper/artifacts/table_w_control_vs_persona_seed1-4_mean_std_20260209.csv`
  - Artifact (CSV): `docs/paper/artifacts/table_w_effect_delta_seed1-4_20260209.csv`
  - Figure (SVG): `docs/paper/figures/table_w_effect_delta_seed1-4_20260209.svg`
  - Regeneration:
    1) `python3 scripts/make_table_w_control_vs_persona.py` (build/refresh Table W + delta CSVs under `docs/paper/artifacts/`)
    2) `python3 scripts/make_paper_figures_from_artifacts.py` (render/overwrite the SVG figure under `docs/paper/figures/`)

**Interpretation check**
- Ensure the Neutral Re-asking Control is described as a *drift baseline* (not adversarial evidence injection).

---

## C3 (Robustness vs recovery): Recovery after flipping is distinct and measurable; interventions can change it

**Where stated**
- `docs/paper/PAPER_DRAFT_EN.md` → §1.4 Core claims (C3)

**Primary evidence**
- Recovery conditional on flip (persona-wise):
  - Artifact (CSV): `docs/paper/artifacts/recovery_personawise_control_vs_persona_seed1-4_mean_std_20260209.csv`
  - Figure (SVG): `docs/paper/figures/recovery_personawise_delta_seed1-4_20260209.svg`
  - Regeneration: `python3 scripts/make_paper_figures_from_artifacts.py`
- Intervention ablation (verify_then_answer vs baseline):
  - Artifact (CSV): `docs/paper/artifacts/recovery_variant_verify_then_answer_vs_baseline_seed1-2_20260210.csv`
  - Paper SSOT run aliases (auditable exports): `results_paper/qwen_vta_seed{1,2}`
  - Regeneration notes:
    - The comparison is computed from `paper_exports/recovery_accuracy.csv` (collapsed over tasks) from the baseline paper SSOT runs vs the verify_then_answer aliases above.
    - (optional figure rendering, if plotted) `python3 scripts/make_paper_figures_from_artifacts.py` (preferred: renders from tracked artifacts only)

---

## C4 (Cross-family): Effects replicate across model families under the same protocol

**Where stated**
- `docs/paper/PAPER_DRAFT_EN.md` → §7.4 Cross-task / cross-family generalization (if included in main)
- `docs/paper/PAPER_DRAFT_EN.md` → §1 Abstract / intro summary (brief cross-family mention)

**Primary evidence (paper-facing)**
- Cross-family visualization (control vs strong persona):
  - Figure (SVG): `docs/paper/figures/cross_family_survival_r5_control_vs_logicaltrap_seed1-2_20260219.svg`
  - Regeneration: `python3 scripts/make_cross_family_figure_svg.py` (reads `docs/paper/artifacts/*`)
- Family-wise survival summaries (CSV → figure):
  - Artifact (CSV): `docs/paper/artifacts/tier1_mistral7b_seed1-2_survival_summary_20260210.csv`
  - Artifact (CSV): `docs/paper/artifacts/tier1_mistralnemo_seed1-2_survival_summary_20260217.csv`
  - Artifact (CSV): `docs/paper/artifacts/tier1_llama3_8b_seed1-2_survival_summary_20260210.csv`
  - Artifact (CSV): `docs/paper/artifacts/tier1_llama3_3b_seed1-2_survival_summary_20260212.csv`
  - Artifact (CSV): `docs/paper/artifacts/tier1_phi3mini_seed1-2_survival_summary_20260217.csv`
  - Artifact (CSV): `docs/paper/artifacts/tier1_zephyr7b_seed1-2_survival_summary_20260218.csv`
  - Regeneration (when `results_paper/` is synced locally; not tracked in git):
    - `python3 scripts/make_tier1_survival_summary.py --run_roots <run1>,<run2> --out_csv <artifact.csv>`
    - For Llama-3.2-3B seeds 1–2 specifically:
      - `results_paper/tier1_llama3_3b_seed1_20260212_030426/`
      - `results_paper/tier1_llama3_3b_seed2_20260212_042339/`

**Audit hook (when the paper bundle is present locally)**
- If you have a local copy of the paper SSOT results directory (often `results_paper/`, *not tracked in git*), run:
  - `python3 scripts/validate_paper_exports.py --results_root results_paper --check_runner_parity`
  - (optional) inspect: `results_paper/GLOBAL_VALIDATE.log`

---

## Sensitivity check (decoding): gaps are qualitatively stable under sampling

**Where stated**
- `docs/paper/PAPER_DRAFT_EN.md` → Abstract + Appendix~A.1 (Decoding sensitivity)

**Primary evidence**
- Decoding sweep (Qwen; temp 0.0 vs 0.7; seeds 1–2):
  - Artifact (CSV): `docs/paper/artifacts/decoding_sweep_qwen_temp_summary_seed1-2_20260211.csv`
  - Figure (SVG): `docs/paper/figures/decoding_sweep_qwen_delta_seed1-2_20260211.svg`
  - Regeneration: `python3 scripts/make_decoding_sweep_figure_svg.py`

---

## Flip interpretation caveat (anti-overclaim): taxonomy is diagnostic, not a metric

**Why this exists**
- Reviewers may over-read flip/TOF rates as direct evidence of *semantic belief change*, especially under strict EM in extractive QA.

**What we claim (and what we do not)**
- Primary claims remain on evaluator-defined **survival / TOF / recovery**.
- Appendix~A.2 provides **diagnostic** interpretation buckets for flips: **boundary/overanswer**, **partial-overlap**, **semantic-change**, plus rare **format/extraction artifacts**.

**Where stated**
- `docs/paper/PAPER_DRAFT_EN.md` → Results intro + Appendix~A.2

---

## Minimal reproducibility checklist (local repo)

- Artifacts are tracked under `docs/paper/artifacts/` (CSV) and rendered figures under `docs/paper/figures/` (SVG).
- **Preferred figure pipeline entrypoint (artifact → SVG, no experiments):**
  - `python3 scripts/make_paper_figures_from_artifacts.py`
  - This script reads **only** the tracked CSV artifacts under `docs/paper/artifacts/` and overwrites the corresponding SVGs under `docs/paper/figures/` (safe/idempotent).
- Any new claim added to Abstract/Intro should be added here with at least one artifact + script pointer (so reviewers can trace claim → CSV → figure).
