# GALILEO EMNLP Main — Heartbeat log & next plan

> Goal: avoid repeating the same shallow update every 10 minutes.
>
> This file is the **single source of truth** for:
> - what changed recently (paper/code/experiments)
> - what we learned / what is still missing
> - what the **next heartbeat** will do (one concrete step)
>
> Update this file at the end of every heartbeat.
>
> IMPORTANT: 매 heartbeat마다 **반드시** `docs/paper/STATUS.md`를 먼저 읽고, 종료 시에는 STATUS를 "리뉴얼(rolling update)" 한다. (중복 보고 방지)

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

## Latest heartbeat notes (append-only)

### 2026-02-09 (pm) — Table W seed1–2 aggregate + artifacts committed

- Completed seed2 pair (auditable green):
  - Control: `results/c2run_control_seed2_20260209_194621/` → validator `[OK]`
  - Persona: `results/c2run_persona_seed2_20260209_200611/` → validator `[OK]`
- Generated Table W per-seed CSVs + seed1–2 mean±std aggregate and **tracked them** under `docs/paper/artifacts/`.
- Updated `PAPER_DRAFT_EN.md` Table W AUTO block to the seed1–2 aggregate (with run roots + artifact pointers).

Next:
- Add seed3 to stabilize std and upgrade the block to seed1–3.

---

### 2026-02-09 (pm) — Fresh control+persona pair complete (auditable pipeline)

- Control run finished end-to-end green under the new auditable pipeline:
  - `results/c2run_control_20260209_172640/`
  - Produced `paper_exports/{survival_curve.csv,turn_of_failure.csv,flip_samples.csv,metadata.json,runner_metadata.json}`
  - Validator: `[OK] results/.../paper_exports` and `[OK] runner_metadata parity`
- Persona run finished end-to-end green under the same pipeline:
  - `results/c2run_persona_20260209_174640/`
  - Produced `paper_exports/{survival_curve.csv,turn_of_failure.csv,flip_samples.csv,metadata.json,runner_metadata.json}`
  - Validator: `[OK] results/.../paper_exports` and `[OK] runner_metadata parity`

What this unlocks:
- We can now refresh Table W (control vs persona) using two run roots that are fully auditable (exports + metadata + runner metadata + validator OK).

Next (paper-facing):
- Generate a paper-consumable summary table (csv/md) and link the run roots + rationale in `docs/paper/PAPER_RESULTS_ANALYSIS_KO.md`.

---

### 2026-02-09 (pm) — Experiment orchestration debugging on nlp8

- Verified the `galileo` conda env is healthy on nlp8 (Py3.11, torch w/ CUDA, vLLM installed).
- Attempted 2 smoke runs (`results/smoke_...`, `results/smoke2_...`) but both produced only a start line in `run.log` and no `paper_exports/`.
  - Root cause hypothesis: orchestration issues from (i) `conda run` / activate hooks and (ii) long commands via `tmux send-keys` being fragile, leading to no visible logging/progress.
- Decision: switch to a **script-based runner** (write `run_smoke.sh` then `bash run_smoke.sh` inside tmux) and avoid `conda run`.

What improved:
- We now have a concrete diagnosis and a robust execution plan; env itself is not the blocker.

What’s still missing:
- A single end-to-end “green” run that emits `paper_exports/*` + metadata + validation logs.

Update:
- ✅ Achieved on nlp8: `results/smoke_20260209_162417/` produced `paper_exports/{survival_curve,turn_of_failure,flip_samples,metadata}.csv/json` + `runner_metadata.json` and validator printed `[OK]`.

---

## 2026-02-09 — Process stabilization + paper-facing Results text landed

- Process guardrails (SSOT): added `docs/paper/HEARTBEAT_PROMPT.md` and `docs/paper/HEARTBEAT_CHECKLIST.md` to prevent drift (wrong server, lane starvation, missing git hygiene).
- Continuity: `docs/paper/STATUS.md` rewritten into **NOW / RECENTLY DONE / TOP GAPS / NEXT HEARTBEAT** format.
- Related work (vault + draft): upgraded TRUTH DECAY + Challenging the Evaluator notes to paper-level protocol and tightened the draft positioning sentences.
- Table W (seed1–4): updated draft summary to seed1–4 and wrote a **Results paragraph** that cites tracked artifacts (Survival@5 drop; Fail@1 increase).

Commits (high-signal):
- `fa8efdd` add SSOT heartbeat prompt
- `681d183` add heartbeat checklist guardrails
- `8cad2f1` STATUS NOW/RECENT/GAPS/NEXT format
- `ba3930b` Table W Results paragraph from tracked artifacts

---

## Next heartbeat plan (ONE step)

**Write one more Results paragraph from paper-ready artifacts (TOF/recovery), or—if artifacts are missing—create the minimal tracked artifact needed and cite it.**

- Preferred: turn an existing export into draft prose (artifact → sentence).
- Output must be a concrete edit in `docs/paper/PAPER_DRAFT_EN.md` + commit.

---
