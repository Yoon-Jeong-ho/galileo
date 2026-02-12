# Claim → Evidence Map (GALILEO EMNLP Main)

Purpose: make the paper *reviewer-auditable* by mapping each core claim to:
1) where it is stated in the draft,
2) what figure/table substantiates it (artifact path), and
3) what script regenerates that artifact.

This file is intentionally short and pragmatic (SSOT for “what proves what”).

---

## C1 (Dynamics): Robustness under pressure is a trajectory (survival + TOF), not a single number

**Where stated**
- `docs/paper/PAPER_DRAFT_EN.md` → §1.4 Core claims (C1)

**Primary evidence (paper-facing)**
- Survival curves across rounds (persona vs control):
  - Artifact (CSV): `docs/paper/artifacts/survival_curve_personawise_control_vs_persona_seed1-4_mean_std_20260209.csv`
  - Figure (SVG): `docs/paper/figures/survival_curves_rounds.svg`
  - Regeneration: `scripts/make_figures_svg.py` (reads `docs/paper/artifacts/*`)
- Early-turn failure / TOF deltas:
  - Artifact (CSV): `docs/paper/artifacts/table_w_effect_delta_seed1-4_20260209.csv`
  - Figure (SVG): `docs/paper/figures/tof_delta_fail1.svg`
  - Regeneration: `scripts/make_figures_svg.py`

**Sanity / audit hooks**
- Paper exports per run (required for auditable aggregation): `paper_exports/` + `metadata.json` + `runner_metadata.json`
- Validator: `scripts/validate_paper_exports.py`

---

## C2 (Mechanism vs drift): Persona pressure induces failures beyond generic multi-turn drift

**Where stated**
- `docs/paper/PAPER_DRAFT_EN.md` → §1.4 Core claims (C2)

**Primary evidence**
- Table W (control vs persona) + effect deltas (ΔSurvival@5, ΔFail@1, etc.):
  - Artifact (CSV): `docs/paper/artifacts/table_w_control_vs_persona_seed1-4_mean_std_20260209.csv`
  - Artifact (CSV): `docs/paper/artifacts/table_w_effect_delta_seed1-4_20260209.csv`
  - Figure (SVG): `docs/paper/figures/tablew_effect_deltas.svg`
  - Regeneration:
    - `scripts/make_table_w_control_vs_persona.py` (build Table W + deltas)
    - `scripts/make_figures_svg.py` (render SVG)

**Interpretation check**
- Ensure the Neutral Re-asking Control is described as a *drift baseline* (not adversarial evidence injection).

---

## C3 (Robustness vs recovery): Recovery after flipping is distinct and measurable; interventions can change it

**Where stated**
- `docs/paper/PAPER_DRAFT_EN.md` → §1.4 Core claims (C3)

**Primary evidence**
- Recovery conditional on flip (persona-wise):
  - Artifact (CSV): `docs/paper/artifacts/recovery_personawise_control_vs_persona_seed1-4_mean_std_20260209.csv`
  - Figure (SVG): `docs/paper/figures/recovery_delta.svg`
  - Regeneration: `scripts/make_figures_svg.py`
- Intervention ablation (verify_then_answer vs baseline):
  - Artifact (CSV): `docs/paper/artifacts/recovery_variant_verify_then_answer_vs_baseline_seed1-2_20260210.csv`
  - Regeneration: `scripts/make_figures_svg.py` (if plotted) + the run exports used to compute it

---

## Minimal reproducibility checklist (local repo)

- Artifacts are tracked under `docs/paper/artifacts/` (CSV) and rendered figures under `docs/paper/figures/` (SVG).
- The figure pipeline entrypoint is typically one of:
  - `scripts/make_figures_svg.py`
  - `scripts/make_paper_figures_from_artifacts.py`
- Any new claim added to Abstract/Intro should be added here with at least one artifact + script pointer.
