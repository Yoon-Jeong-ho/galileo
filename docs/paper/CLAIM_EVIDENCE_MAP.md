# Claim → Evidence Map (GALILEO EMNLP Main)

Purpose: keep the paper *reviewer-auditable* by mapping each core claim to:
1) where it is stated in the **LaTeX main draft**,
2) which figure/table substantiates it (repo artifact path), and
3) which script regenerates that artifact.

This file is intentionally pragmatic: a SSOT for “what proves what”, and a guardrail against label/filename drift.

---

## Terminology guardrails (avoid reviewer confusion)

- **Survival(r)**: probability the model remains correct **for every round 1..r** (cumulative), *not* “accuracy at round r only.”
- **Flip**: a trajectory event **correct → incorrect** between rounds.
- **TTF (time-to-first-failure; older drafts: TOF)**: the **first challenged round index** where a flip occurs; **Fail(1)** is the probability that TTF = 1.
- **Recovery(flip)**: correctness **after** a flip (distinct from survival/TTF); reported separately.

### Conditioning-set reporting modes (must be explicit in captions)

We use two matched-set modes; the draft must say which one is used for each plot/table:

- **persona-matched (within-persona attribution)**: for each persona arm we filter to that persona’s initially-correct subset `C_p` (Phase 1), then run both (i) persona pressure and (ii) the paired **Neutral Re-asking Control (NRC_p)** on that same `C_p`. This makes persona-vs-control comparisons apples-to-apples *within* persona, but **NRC values can differ across personas** because denominators `C_p` differ.

- **shared-C (cross-persona comparability)**: define a single initially-correct set `C` under a persona-free neutral prompt, then evaluate *every* persona arm and NRC on exactly the same `C`. This is best when directly comparing personas on the same denominator.

---

## LaTeX label ↔ repo file mapping (anti-drift)

**Source of truth:** `docs/paper/latex_paper_emnlp2023/main.tex`.

### Figures used in the main draft

- `fig:protocol` → `docs/paper/figures/protocol_overview.svg`
- `fig:survival` → `docs/paper/figures/survival_curves_rounds_seed1-4_20260209.svg`
- `fig:fail1` → `docs/paper/figures/tof_personawise_fail1_delta_seed1-4_20260209.svg`
- `fig:crossfam` → `docs/paper/figures/cross_family_survival_r5_control_vs_logicaltrap_seed1-2_20260221.svg` (canonical; keep LaTeX + this map aligned)

### Appendix figures referenced in the main draft text

- `fig:recovery` → `docs/paper/figures/recovery_personawise_delta_seed1-4_20260209.svg`
- `fig:decoding` → `docs/paper/figures/decoding_sweep_qwen_delta_seed1-2_20260211.svg`

### Tables

- `tab:main` → Table rows come from:
  - `docs/paper/latex_paper_emnlp2023/generated/table1_rows.tex` (if present) or
  - `docs/paper/latex_paper_emnlp2023/static/table1_rows.tex` (fallback tracked in git)

  The numeric inputs for absolute Fail@1 and Recovery@flip are sourced from the latest tracked artifact:
  - `docs/paper/artifacts/table1_from_results_paper_exports_*.csv`

  This artifact is generated from paper-ready runs staged under `results_paper/` on the experiment SSOT machine (nlp8).

(If any filename/label changes due to regeneration, update **both** LaTeX and this map in the same commit.)

---

## Abstract/Intro: claim-level proof pointers (what a reviewer will notice first)

These are the claims most likely to be read without the Appendix. Each needs an obvious proof pointer.

**A0 (motivation): multi-turn pressure can induce deviations not visible in single-turn accuracy.**
- Where stated: `main.tex` Abstract + Intro (§1)
- Evidence:
  - Dynamics: `fig:survival` → `docs/paper/figures/survival_curves_rounds_seed1-4_20260209.svg`
  - Early vulnerability: `fig:fail1` → `docs/paper/figures/tof_personawise_fail1_delta_seed1-4_20260209.svg`

**A1 (protocol): GALILEO + matched drift baseline (NRC).**
- Where stated: `main.tex` Abstract + Protocol (§4)
- Evidence:
  - Protocol overview: `fig:protocol` → `docs/paper/figures/protocol_overview.svg`
  - Drift-corrected reporting is reflected throughout Results as persona–NRC deltas (e.g., `tab:main`).

**A2 (metrics): survival / TTF+Fail(1) (older drafts: TOF) / Recovery(flip) are distinct.**
- Where stated: `main.tex` Intro + Metrics/Results
- Evidence:
  - Survival: `fig:survival`
  - Fail(1): `fig:fail1`
  - Recovery: `fig:recovery` (Appendix figure referenced from main text)

**A3 (replication): effects persist across open-weight families (at least seeds 1–2).**
- Where stated: `main.tex` Results (§5.3)
- Evidence:
  - Cross-family barplot: `fig:crossfam` → `docs/paper/figures/cross_family_survival_r5_control_vs_logicaltrap_seed1-2_20260221.svg`
  - Main-table cross-family aggregates (Survival@5 / Fail@1 / Recovery@flip): `tab:main`
    - Latest tracked Table-1 artifact: `docs/paper/artifacts/table1_from_results_paper_exports_20260226_0040.csv`
    - Mean±std helper note (for the newly added Llama-3.2-3B + Nemo seed1–2):
      `docs/paper/artifacts/table1_from_results_paper_exports_20260226_0040_agg_mean_std.txt`

---

## Regeneration entry points (scripts)

- Paper-facing figure rendering from tracked CSV artifacts:
  - `python3 scripts/make_paper_figures_from_artifacts.py`

- Cross-family figure:
  - `python3 scripts/make_cross_family_figure_svg.py`

- Table 1 (`tab:main`) row generation (if using generated rows):
  - `python3 scripts/make_latex_table1_rows.py` (if/when present; otherwise keep `static/table1_rows.tex` updated)

---

## Reviewer-risk guardrails (common failure modes)

- **Always state the denominator mode (shared-C vs persona-matched C_p)** in each caption that compares personas.
- **Delta-first aggregation:** whenever we report pooled persona–NRC effects, compute deltas within each persona pairing before averaging (avoids mixing unmatched controls).
- **Recovery denominators:** Recovery(flip) is undefined when a track has no flips; captions/tables must either (i) show denominators or (ii) state the renormalization rule used when excluding undefined cells.
