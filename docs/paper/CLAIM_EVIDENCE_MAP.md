# Claim → Evidence Map (GALILEO EMNLP Main)

Purpose: keep the paper *reviewer-auditable* by mapping each core claim to:
1) where it is stated in the **LaTeX main draft**,
2) which figure/table substantiates it (repo artifact path), and
3) which script regenerates that artifact.

This file is intentionally pragmatic: a SSOT for “what proves what”, and a guardrail against label/filename drift.

---

## Active local baseline (2026-03-10)

These are the currently verified runs under `/data_x/aa007878/projects/galileo`, and they take precedence over older heartbeat/status notes when there is a conflict.

- Synthetic smoke:
  - `/data_x/aa007878/projects/galileo/tmp/results/smoke_gpu5_20260310_184715/`
- Math pilot (small):
  - `/data_x/aa007878/projects/galileo/tmp/results/pilot_gpu5_real_20260310_185233/`
- Math pilot (50-sample baseline):
  - `/data_x/aa007878/projects/galileo/tmp/results/pilot50_gpu5_20260310_185825/`
- Non-math sanity (ARC-Easy MCQA):
  - `/data_x/aa007878/projects/galileo/tmp/results/sanity_arc_gpu6_20260310_191641/`
- Evidence-gate mitigation mains:
  - `/data_x/aa007878/projects/galileo/tmp/results/qwen7b_gsm8k_control_authority_evidencegate_gpu5_20260310_230240/`
  - `/data_x/aa007878/projects/galileo/tmp/results/qwen7b_arc_control_authority_evidencegate_gpu5_20260310_230240/`

All of the above have `paper_exports/` plus validator `[OK]`. They are baseline evidence for:
- pressure (`authority_claim`) < control (`control_reask`) on the same protocol
- evidence-bearing recovery path executing end-to-end
- preliminary mitigation (`evidence_gate`) being measurable within the same export schema

Current stronger-vs-weaker claim split:

- **Strong (local multiseed-backed):**
  - evidence baseline on GSM8K + ARC-Easy (`seed 1–3`)
  - artifact paths:
    - `/data_x/aa007878/projects/galileo/docs/paper/artifacts/qwen7b_evidence_multiseed_seed1-3_metrics_20260310.csv`
    - `/data_x/aa007878/projects/galileo/docs/paper/artifacts/qwen7b_evidence_multiseed_seed1-3_deltas_20260310.csv`
- **Strong (local multiseed-backed, correction baseline):**
  - grounded baseline on GSM8K + ARC-Easy (`seed 1–3`)
  - artifact paths:
    - `/data_x/aa007878/projects/galileo/docs/paper/artifacts/qwen7b_grounded_multiseed_seed1-3_metrics_20260310.csv`
    - `/data_x/aa007878/projects/galileo/docs/paper/artifacts/qwen7b_grounded_multiseed_seed1-3_deltas_20260310.csv`
- **Provisional but stronger than seed-1 (local multiseed-backed mitigation):**
  - evidence_gate mitigation on GSM8K + ARC-Easy (`seed 1–3`)
  - artifact paths:
    - `/data_x/aa007878/projects/galileo/docs/paper/artifacts/qwen7b_evidencegate_multiseed_seed1-3_metrics_20260310.csv`
    - `/data_x/aa007878/projects/galileo/docs/paper/artifacts/qwen7b_evidencegate_multiseed_seed1-3_deltas_20260310.csv`
    - `/data_x/aa007878/projects/galileo/docs/paper/artifacts/qwen7b_threeway_multiseed_comparison_20260310.csv`
    - `/data_x/aa007878/projects/galileo/docs/paper/artifacts/qwen7b_tradeoff_gsm8k_multiseed_20260310.svg`
    - `/data_x/aa007878/projects/galileo/docs/paper/artifacts/qwen7b_tradeoff_arc_multiseed_20260310.svg`
  - still interpret as **promising mitigation with trade-offs**, not a final headline solution

- **Qualitative support (partial manual labeling, provisional):**
  - `/data_x/aa007878/projects/galileo/tmp/analysis/manual_flip_taxonomy_partial_20260310.csv`
  - `/data_x/aa007878/projects/galileo/tmp/analysis/manual_taxonomy_guidelines_20260310.md`

Do **not** treat these as final headline claims yet; they are the active baseline for expanding to larger math + non-math evidence sets.

### Verified March 10, 2026 Qwen7B multiseed proof bundle

These are the strongest currently verified **same-model / same-protocol** proof pointers in the repo and should be preferred over older seed-1-only notes:

- **Evidence-bearing baseline multiseed**
  - root: `/data_x/aa007878/projects/galileo/tmp/results/qwen7b_evidence_multiseed_gpu5_20260310_231212/`
  - validator: `/data_x/aa007878/projects/galileo/tmp/results/qwen7b_evidence_multiseed_gpu5_20260310_231212/GLOBAL_VALIDATE.log`
  - tracked CSVs:
    - `/data_x/aa007878/projects/galileo/docs/paper/artifacts/qwen7b_evidence_multiseed_seed1-3_metrics_20260310.csv`
    - `/data_x/aa007878/projects/galileo/docs/paper/artifacts/qwen7b_evidence_multiseed_seed1-3_deltas_20260310.csv`
- **Grounded-correction baseline multiseed**
  - root: `/data_x/aa007878/projects/galileo/tmp/results/qwen7b_grounded_multiseed_gpu5_20260310_232747/`
  - validator: `/data_x/aa007878/projects/galileo/tmp/results/qwen7b_grounded_multiseed_gpu5_20260310_232747/GLOBAL_VALIDATE.log`
  - tracked CSVs:
    - `/data_x/aa007878/projects/galileo/docs/paper/artifacts/qwen7b_grounded_multiseed_seed1-3_metrics_20260310.csv`
    - `/data_x/aa007878/projects/galileo/docs/paper/artifacts/qwen7b_grounded_multiseed_seed1-3_deltas_20260310.csv`
- **Evidence-gate mitigation multiseed**
  - root: `/data_x/aa007878/projects/galileo/tmp/results/qwen7b_evidencegate_multiseed_gpu5_20260310_234316/`
  - validator: `/data_x/aa007878/projects/galileo/tmp/results/qwen7b_evidencegate_multiseed_gpu5_20260310_234316/GLOBAL_VALIDATE.log`
  - tracked CSVs / figures:
    - `/data_x/aa007878/projects/galileo/docs/paper/artifacts/qwen7b_evidencegate_multiseed_seed1-3_metrics_20260310.csv`
    - `/data_x/aa007878/projects/galileo/docs/paper/artifacts/qwen7b_evidencegate_multiseed_seed1-3_deltas_20260310.csv`
    - `/data_x/aa007878/projects/galileo/docs/paper/artifacts/qwen7b_threeway_multiseed_comparison_20260310.csv`
    - `/data_x/aa007878/projects/galileo/docs/paper/artifacts/qwen7b_tradeoff_gsm8k_multiseed_20260310.svg`
    - `/data_x/aa007878/projects/galileo/docs/paper/artifacts/qwen7b_tradeoff_arc_multiseed_20260310.svg`

Reviewer-safe takeaways supported directly by those files:

- pressure still reduces Survival@5 relative to control under both evidence-bearing and grounded-correction baselines;
- the effect is larger on ARC-Easy than on GSM8K in the current Qwen7B multiseed package;
- evidence-gate is measurable in the same export schema, but should still be framed as a trade-off, not a final mitigation headline.

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
    - Latest tracked Table-1 artifact: `docs/paper/artifacts/table1_from_results_paper_exports_20260227_2038.csv` (includes Mistral‑7B seed1)
    - Mean±std helper note (seed-level provenance retained; aggregation helper):
      `docs/paper/artifacts/table1_from_results_paper_exports_20260227_2038_agg_mean_std.txt`

**A4 (same-model robustness persists across correction baselines): Qwen7B still shows authority-pressure degradation under multiseed evidence-bearing and grounded-correction evaluation.**
- Where stated:
  - currently explicit in `README.md` §1.1 (“현재까지 한 일 (요약)” / 2026-03-10 multiseed bullets)
  - **not yet pinned to a main-draft section**; keep this as repo-level evidence until draft text is added
- Evidence:
  - Evidence-bearing multiseed:
    - `docs/paper/artifacts/qwen7b_evidence_multiseed_seed1-3_metrics_20260310.csv`
    - `docs/paper/artifacts/qwen7b_evidence_multiseed_seed1-3_deltas_20260310.csv`
  - Grounded-correction multiseed:
    - `docs/paper/artifacts/qwen7b_grounded_multiseed_seed1-3_metrics_20260310.csv`
    - `docs/paper/artifacts/qwen7b_grounded_multiseed_seed1-3_deltas_20260310.csv`
  - Concrete examples already verified in the tracked CSVs:
    - grounded GSM8K ΔSurvival@5 = `-0.208467`
    - grounded ARC-Easy ΔSurvival@5 = `-0.612267`

**A5 (mitigation trade-off): evidence-gate changes the same-model trade-off surface, but it is not yet a clean headline mitigation claim.**
- Where stated:
  - currently explicit in `README.md` §1.1 (`evidence_gate mitigation (2026-03-10, seeded main)` and the multiseed caution note)
  - **not yet pinned to a main-draft section**; do not promote beyond repo-level trade-off evidence until draft text exists
- Evidence:
  - `docs/paper/artifacts/qwen7b_evidencegate_multiseed_seed1-3_metrics_20260310.csv`
  - `docs/paper/artifacts/qwen7b_evidencegate_multiseed_seed1-3_deltas_20260310.csv`
  - `docs/paper/artifacts/qwen7b_threeway_multiseed_comparison_20260310.csv`
  - `docs/paper/artifacts/qwen7b_tradeoff_gsm8k_multiseed_20260310.svg`
  - `docs/paper/artifacts/qwen7b_tradeoff_arc_multiseed_20260310.svg`
- Guardrail:
  - cite these as **trade-off evidence** unless/until a broader, cleaner evidence set is verified.
  - `scripts/aggregate_condition_multiseed.py` covers the multiseed metrics/deltas only; the three-way CSV and tradeoff SVGs are currently **tracked outputs whose regeneration path still needs to be pinned** before they should back any headline claim.

---

## Regeneration entry points (scripts)

- Paper-facing figure rendering from tracked CSV artifacts:
  - `python3 scripts/make_paper_figures_from_artifacts.py`

- Condition multiseed metric/delta aggregation:
  - `python3 scripts/aggregate_condition_multiseed.py --results_root <multiseed_root> --out_dir <out_dir>`

- Cross-family figure:
  - `python3 scripts/make_cross_family_figure_svg.py`

- Table 1 (`tab:main`) row generation (if using generated rows):
  - `python3 scripts/make_latex_table1_rows.py` (if/when present; otherwise keep `static/table1_rows.tex` updated)

---

## Reviewer-risk guardrails (common failure modes)

- **Always state the denominator mode (shared-C vs persona-matched C_p)** in each caption that compares personas.
- **Delta-first aggregation:** whenever we report pooled persona–NRC effects, compute deltas within each persona pairing before averaging (avoids mixing unmatched controls).
- **Recovery denominators:** Recovery(flip) is undefined when a track has no flips; captions/tables must either (i) show denominators or (ii) state the renormalization rule used when excluding undefined cells.
