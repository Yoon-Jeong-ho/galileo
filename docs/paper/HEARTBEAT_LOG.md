# GALILEO EMNLP Main — Heartbeat log & next plan

> Goal: avoid repeating the same shallow update every 10 minutes.
>
> This file is the **single source of truth** for:
> - what changed recently (paper/code/experiments)
> - what we learned / what is still missing
> - what the **next heartbeat** will do (one concrete step)
>
> Update this file at the end of every heartbeat.

---

## Current top priorities (ordered)

1) **Paper: lock the novelty story vs prior multi-turn sycophancy work**
   - Must explicitly position vs SYCON / TRUTH DECAY / rebuttal-framing work.
   - Must keep the *ground-truth + dynamics + recovery + drift control* delta crisp.

2) **Experiments: re-enable a reliable, auditable pipeline**
   - Ensure every run produces: `paper_exports/*`, `metadata.json`, `runner_metadata.json`, `GLOBAL_VALIDATE.log`.
   - Ensure the drift baseline is standardized as `neutral_reask_control` in exports.

3) **Planning discipline**
   - Each heartbeat does exactly ONE high-impact action and records it here.

---

## What happened recently (2026-02-09)

### Paper / positioning

- Standardized the drift baseline naming in the paper as **Neutral Re-asking Control**.
- Added multi-turn sycophancy related work notes (KO) covering:
  - SYCON Bench (Turn of Flip / Number of Flip)
  - TRUTH DECAY benchmark
  - Challenging the Evaluator (follow-up rebuttal vs evaluator framing)

**Repo commits (high-level):**
- `0121d46`: add multi-turn sycophancy related work notes.
- `df647e0`: note PR-based workflow for Codex code review.

### Code / exports / reproducibility

- `53fe655`: paper export now normalizes the control label (e.g., "Control Re-asking") to `neutral_reask_control`.
- `cdfa8b6`: validator `--require_control` applies only to control export bundles.
- Added/expanded metadata + validation scripts and wired them into runner scripts (see git log around `3df3cf8`..`46513ea`).

### Remote (nlp8)

- SSH BatchMode access to nlp8 was fixed (passphrase + known_hosts issue).
- Identified existing results directory: `results/nlp8_control_vs_persona_20260209_022414/`.
- Re-exported **control seed1** with the new exporter to show the control id becomes `neutral_reask_control` and `metadata.json` appears.
- Remaining gap: legacy runs lack `runner_metadata.json` unless rerun under new runners.

---

## What is still missing / risks

1) **Runs are not yet fully “auditable green”**
   - Legacy results lack `runner_metadata.json` and `GLOBAL_VALIDATE.log`.
   - Need at least one fresh run under the new runner scripts to prove end-to-end.

2) **Control identifier consistency in *all* exports**
   - Control bundles should export `neutral_reask_control` (now fixed in exporter), but old exports still need rerun/re-export.

3) **Paper integration (EN draft)**
   - Related Work in `PAPER_DRAFT_EN.md` should explicitly map SYCON’s Turn of Flip ↔ our TOF and cite TRUTH DECAY + rebuttal framing.

---

## Next heartbeat plan (ONE step)

**Plan A (preferred): run a smoke experiment on nlp8 (GPU 0 only)**

- Objective: produce a brand-new run that generates:
  - `paper_exports/*`
  - `metadata.json`
  - `runner_metadata.json`
  - `GLOBAL_VALIDATE.log`
- Constraints:
  - Use GPU **0** only (`CUDA_VISIBLE_DEVICES=0`).
  - Use `tmux`.
  - Keep it light: small `NUM_SAMPLES` (e.g., 50–100) and TP=1.

If the smoke run is green, subsequent heartbeats can scale to multi-seed/family.

**Plan B (if remote blocked): integrate the new related work into `PAPER_DRAFT_EN.md`**

---
