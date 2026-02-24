# Main Table + Figures plan (reviewer-first)

Purpose: eliminate the “no main table / vague figures / uneven section lengths” failure mode by locking a reviewer-first *presentation* skeleton.

This file is a **presentation SSOT** (what the paper should show). Numerical content must remain sourced from `docs/paper/artifacts/` and `results_paper/`.

---

## 1) Main table (Table 1) — one-stop summary

### Table 1 goal
A single table that (i) defines the core evaluation objects and (ii) reports the primary results with minimal hunting.

**Rows (minimum):**
- Model families (Tier‑1, seeds 1–2 unless stated):
  - Qwen2.5‑7B‑Instruct (seed1–4 for main story)
  - Llama‑3.1‑8B‑Instruct (seed1–2)
  - Mistral‑7B‑Instruct (seed1–2)
  - Llama‑3.2‑3B‑Instruct (seed1–2)
  - Phi‑3‑mini‑4k‑instruct (seed1–2)
  - Mistral‑Nemo (seed1–2)
  - Qwen2.5‑14B‑Instruct (seed1–2)

**Columns (primary metrics):**
- Survival@r (r=5): **C vs P**
- Fail@1 (or TTF summary; older drafts: TOF): **C vs P**
- Recovery(flip): **C vs P**
- Δ for each metric, with fixed sign convention: **Δ = P − C** on the matched initially-correct set
- `n0/seed`: average number of initially-correct examples per seed for that model row (pooled across tasks/personas); prevents weighting ambiguity
- #seeds / #samples / #tasks (small footnote)

**Formatting rules:**
- 1 row per (model family), aggregated over seeds.
- Put the *persona-weighted aggregate* as the primary number; show mean±std across seeds.
- Mark which numbers come from `Table W` (control vs persona) artifacts.

**Evidence pointers:**
- Each cell must be reconstructible from:
  - `docs/paper/artifacts/*.csv` (tracked)
  - and the cited paper-ready run roots under `results_paper/` (validator parity OK)

---

## 2) Figures (minimum set)

### Figure 1 — protocol overview (already exists)
- Use `docs/paper/figures/protocol_overview.svg` (source) + PDF for LaTeX.
- Caption emphasizes: (i) ground-truth tasks, (ii) multi-round persona pressure, (iii) Neutral Re-asking Control (NRC), (iv) survival/TTF/recovery definitions (older drafts: TOF).

### Figure 2 — main survival curves (Qwen seed1–4)
- `docs/paper/figures/survival_curves_rounds_seed1-4_20260209.svg`
- Show control vs persona across rounds (one panel, small multiples if needed).

### Figure 3 — persona-wise deltas (Qwen seed1–4)
Pick one of:
- `docs/paper/figures/survival_r5_personawise_delta_seed1-4_20260209.svg`
- or `docs/paper/figures/tof_personawise_fail1_delta_seed1-4_20260209.svg`
- or `docs/paper/figures/recovery_personawise_delta_seed1-4_20260209.svg`

Rule: only one persona-wise delta in main; the other two go to appendix.

### Figure 4 — cross-family generalization (Tier‑1 seeds 1–2)
- `docs/paper/figures/cross_family_survival_r5_control_vs_logicaltrap_seed1-2_20260221.svg`
- Caption must state that protocol/decoding are matched; NRC provides drift baseline.

### Figure 5 — decoding sensitivity sweep (Tier‑1 seeds 1–2)
- `docs/paper/figures/decoding_sweep_qwen_delta_seed1-2_20260211.svg`

---

## 3) Section length budget (8–9 pages target)

Hard constraint: avoid reviewer fatigue.

Suggested budget:
- Introduction: 1.0–1.25p
- Protocol/Threat model: 0.75–1.0p (includes Fig 1)
- Metrics/Definitions: 0.5–0.75p (Survival/TTF/Recovery; keep crisp; older drafts: TOF)
- Experiments: 2.0–2.5p (Table 1 + Figs 2–5)
- Related Work: 0.75–1.0p (tight, proof-pointer style)
- Limitations/Ethics: 0.5–0.75p

---

## 4) Immediate TODOs (for next writing heartbeat)

1) **DONE (2026-02-22):** Wire Table 1 skeleton into `docs/paper/PAPER_DRAFT_EN.md` (LaTeX tabular placeholder + caption with fixed sign conventions + explicit `n0/seed` column).
2) Populate Table 1 row-by-row by linking each cell to a specific tracked artifact under `docs/paper/artifacts/` and a paper-ready run root under `results_paper/`.
   - Use the same notation as the draft: `C` = NRC (control), `P` = persona pressure, `Δ` = `P − C` on the matched initially-correct set.
   - Report each numeric cell as mean±std across seeds; keep persona-weighted aggregates as the default unless explicitly labeled otherwise.
   - Fill `n0/seed` (average initially-correct examples per seed) to make weighting/comparability unambiguous.
3) Ensure every figure referenced in the draft has a corresponding entry in `docs/paper/FIGURE_CAPTIONS.md` with provenance → artifacts.
4) Add 1–2 sentences in Results that explicitly say “Table 1 is the one-stop summary; figures provide decomposition.”
