# GALILEO — EMNLP Main Execution Plan (SSOT)

Last updated: 2026-02-24

This is the **one plan** the 10‑minute heartbeat loop must follow.

## A. What “progress” means (reviewer-risk reducers)

We prioritize work that reduces the highest-probability reviewer objections:

1) **Clarity / navigability**
   - Every major claim has an explicit proof pointer (Table/Figure + artifact path).
   - Acronyms introduced once, then used consistently (e.g., NRC).

2) **Reproducibility / auditability**
   - Every paper number comes from tracked artifacts under `docs/paper/artifacts/`.
   - Every cited run is in `results_paper/` and passes validator + runner-metadata parity.

3) **Generalization / robustness**
   - Tier‑1 cross-family coverage (at least one additional model family with seeds 1–2).
   - Exactly one ablation that strengthens a key claim (only if it changes the story).

## B. 4 lanes (we rotate, but only ONE per heartbeat)

1) **Experiments (Tier‑1)** — remote SSOT (`ssh nlp8`, `/data_x/aa007878/galileo`)
2) **Paper writing** — local SSOT (EN draft / LaTeX)
3) **Paper dev / tooling** — exports, validators, table/figure regeneration
4) **Literature / positioning** — only when 1–3 are stable

## C. Weekly plan (compact)

### C1) Paper structure lock (highest leverage)
- Lock the narrative skeleton:
  - Problem → protocol → metrics → main results (Survival/TOF/Recovery) → cross-family → limitations.
- Enforce **claim→evidence** pointers:
  - SSOT: `docs/paper/CLAIM_EVIDENCE_MAP.md`

### C2) Table/Figure SSOT lock
- **Main table** is auto-filled from artifacts:
  - survival/fail@n0 already tracked; next is Recovery@flip and cross-family coverage.
- Figures regenerate from tracked CSVs:
  - SSOT: `docs/paper/FIGURE_CAPTIONS.md` + `docs/paper/figures/` + `docs/paper/artifacts/`

### C3) Compute plan (Tier‑1 only)
- Don’t “run more”; run only if it reduces a specific reviewer risk.
- Default next experiments (choose one, after GPU preflight):
  1) Add **one more model family** with seeds 1–2 under the same protocol.
  2) Add **one ablation** (e.g., recovery_variant) only if it clarifies mechanism.
  3) More seeds only if CIs look fragile in plots/tables.

## D. How the agent should operate (10-min heartbeat contract)

Each heartbeat must:
1) Read `docs/paper/STATUS.md` and pick the primary lane.
2) Produce one concrete deliverable (diff, artifact, validated export, or drafted paragraph with citations).
3) Update `STATUS.md` + append to `HEARTBEAT_LOG.md`.
4) If any repo changes: commit + push.

## E. Immediate next 3 deliverables (recommended)

1) **Finish Table~1 Recovery@flip auto-fill** into EN draft (remove manual editing risk).
2) **Cross-family figure/table consistency check** (sign conventions, pp vs %, caption language).
3) **Abstract + Intro proof-pointer sweep** (reduce reviewer search cost).
