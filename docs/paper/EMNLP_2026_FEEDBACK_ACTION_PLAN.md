# EMNLP 2026 — Feedback→Action Plan (SSOT) + Heartbeat Checklist

Last updated: 2026-02-24

This file is the **execution SSOT** for “what we do next” based on the latest external feedback bundle.

## 0) One-sentence positioning (must survive reviewer attack)

> **GALILEO isolates pressure-induced answer deviation from generic multi-turn drift** by pairing each pressure trajectory with a matched **Neutral Re-asking Control (NRC)**, conditioned on initially-correct examples, and reporting **persona − NRC deltas** as auditable artifacts.

**Non-negotiable framing:** we are **not** claiming survival analysis or ToF is new; we claim **drift-corrected attribution** + auditability + recovery axis.

## 1) Top reviewer attack points → required counters (from feedback)

### A. “This is MT-Consistency / Time-To-Inconsistency / SYCON again. What’s new?”
**Counter to bake into Related Work + Results:**
- We contribute **matched drift control (NRC)** as a counterfactual baseline to **deconfound** pressure effects.
- We show **rankings change** if you evaluate persona-only vs persona−NRC deltas.

### B. Recovery is missing/‘--’ and the definition is denominator-biased
**Counter:**
- Fill Table 1 recovery coverage (remove “--” where feasible).
- Add **shared-denominator recovery** (see §3.2) so persona vs NRC are compared on the **same flip set**.

### C. Terminology collision: SYCON’s ToF/NoF vs our TOF
**Counter:**
- Rename or explicitly disambiguate: prefer **TTF/TFF = time-to-(first)-failure** in paper.
- Add **NoF** (number of flips) as a secondary instability metric (cheap + aligns with SYCON).

### D. Conditioning bias (“you only test easy items”)
**Counter:**
- Report **conditioning rate** |C|/|D| and Phase-1 accuracy.
- Add an appendix: unconditioned per-round accuracy (informational, not the main claim).

## 2) Deliverables (paper-facing, tracked)

### 2.1 Required artifacts (tracked; must exist)
- [ ] Table 1 complete (Survival/Fail@1/Recovery) from `docs/paper/artifacts/table1_from_results_paper_exports_<date>.csv`.
- [ ] A single “**NRC is necessary**” figure/table:
  - persona-only ranking vs drift-corrected ranking (Spearman + top-k changes).
- [ ] Related-work “direct overlap” paragraph + comparison table (FlipFlop, MT-Consistency/PWC, Time-To-Inconsistency, SYCON, Truth Decay).

### 2.2 Process / reproducibility
- [ ] Every cited run is paper-ready: `paper_exports/` + `runner_metadata.json` + validator OK.
- [ ] Token-cap guardrail: no headline evidence from runs with `max_tokens capped to 1`.

## 3) Experiment plan (priority order)

### 3.1 Tier‑1 completion (Table 1 looks weak → fix first)
Goal: remove “--” by ensuring missing families have **recovery exports** and are staged under `results_paper/`.

- [ ] **Phi‑3‑mini** seed1–2 (standard Tier‑1; includes recovery exports)
- [ ] **Mistral‑Nemo** seed1–2 (standard Tier‑1; includes recovery exports)

Rules:
- SSOT remote: `ssh nlp8`, repo `/data_x/aa007878/galileo`.
- GPUs: 0–6 idle-only + PID→user check + CUDA alloc preflight.

### 3.2 Shared-denominator Recovery (fix the denominator artifact)
Add a new metric:
- Define `F_shared = F_persona ∪ F_NRC` (within a matched pair).
- Compute recovery on **the same F_shared** for both arms.

Deliverable:
- [ ] Tracked artifact CSV + one figure (delta) showing standard Recovery@flip vs shared-denominator recovery.

### 3.3 “NRC matters” ranking flip experiment (mandatory)
Deliverable:
- [ ] A table/figure showing that persona-only rankings can mislead vs drift-corrected deltas.
  - Spearman correlation
  - top-k membership change

### 3.4 Mitigation demo (1–2 only; high ROI)
Pick 1 intervention that is cheap and reviewer-legible.
- [ ] **Third-person / anti-sycophancy system prompt** variant (baseline vs mitigation on same protocol)
- [ ] **verify_then_answer** (already partially exists; extend to one more family if needed)

## 4) Heartbeat checklist (10-min loop)

Every heartbeat must do **exactly one** primary lane and tick items here.

### 4.1 Start-of-heartbeat (always)
- [ ] Read `docs/paper/STATUS.md` and pick primary lane.
- [ ] If running experiments: check `tmux ls` + `nvidia-smi` + PID→user for candidate GPU.

### 4.2 One-lane options (pick ONE)

**Lane: Experiments (nlp8)**
- [ ] Launch exactly one tmux run OR validate/stage one completed run.
- [ ] Ensure `paper_exports/recovery_accuracy.csv` exists (or explicitly note why it cannot).
- [ ] Run validator + token-cap gate.

**Lane: Table/Figure pipeline**
- [ ] Regenerate `table1_from_results_paper_exports_<date>.csv`.
- [ ] Regenerate the needed LaTeX rows / SVG figures from artifacts.

**Lane: Analysis**
- [ ] Add one proof-pointer-backed paragraph to `PAPER_RESULTS_ANALYSIS_KO.md` or EN draft.

**Lane: Related work / positioning**
- [ ] Add one “overlap vs our delta” paragraph + citation + proof pointer.

### 4.3 End-of-heartbeat (always)
- [ ] Update `docs/paper/STATUS.md` (rolling rewrite).
- [ ] Append to `docs/paper/HEARTBEAT_LOG.md` (3–6 lines).
- [ ] If repo changed: commit + push.

## 5) Notes / constraints

- Use all *idle and healthy* GPUs on nlp8 **sequentially** (one heavy run at a time); do not collide with other users.
- If the nlp8 `data/` directory is incomplete vs the paper’s benchmark list, do **not** silently treat a subset as Tier‑1 without labeling it as such.
