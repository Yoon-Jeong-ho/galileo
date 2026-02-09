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

## 2026-02-09 — Results section upgraded to artifact-cited prose (seed1–4)

- Results prose is now backed by **tracked CSV artifacts** for:
  - Survival (persona-wise Survival@5)
  - TOF (collapsed + persona-wise Fail@1/Never-fail)
  - Recovery (collapsed + persona-wise)
  - Control vs persona (Table W + Δ effect-size)
- Added a Sec 7 Results preface that states the **seed1–4 auditable green** convention and points readers to `docs/paper/artifacts/`.

Representative commits:
- `4e29ce8` TOF collapsed artifact + cite
- `1e636e6` TOF persona-wise artifact + cite
- `10b68da` recovery artifacts + cite
- `3783e05` survival@5 persona-wise artifact + cite
- `8394b30` Results preface (seed1–4, tracked artifacts)

---

## Next heartbeat plan (ONE step)

**Add numeric contrast sentences to Related Work (Sec 6.4) by extracting 1–2 effect sizes from TRUTH DECAY and/or Challenging the Evaluator.**

- Deliverable: update one vault note’s results section + add 1–2 citation sentences with concrete numbers in `docs/paper/PAPER_DRAFT_EN.md`.

---

### 2026-02-10 (am) — Submission-ready SVG figures generated from tracked artifacts

- Ran `scripts/make_paper_figures_from_artifacts.py` to generate vector figures directly from tracked CSV artifacts (stdlib-only, reproducible).
- Outputs (seed1–4; all under `docs/paper/figures/`):
  - `survival_curves_rounds_seed1-4_20260209.svg`
  - `survival_r5_personawise_delta_seed1-4_20260209.svg`
  - `tof_personawise_fail1_delta_seed1-4_20260209.svg`
  - `recovery_personawise_delta_seed1-4_20260209.svg`
  - `table_w_effect_delta_seed1-4_20260209.svg`

Next:
- Wire these into `docs/paper/PAPER_DRAFT_EN.md` (captions + in-text callouts) and decide whether to emit PDF variants.

### 2026-02-10 (am) — Wired generated SVG figures into the EN draft (LaTeX include snippets)

- Updated `docs/paper/PAPER_DRAFT_EN.md` Results section to reference the committed SVGs under `docs/paper/figures/`.
- Added copy-pastable LaTeX `figure` snippets + suggested in-text callouts for:
  - survival curves over rounds
  - persona-wise ΔSurvival@5
  - persona-wise ΔFail@1
  - persona-wise ΔRecovery@flip
  - Table W effect deltas

Next:
- Decide SVG vs PDF in the final LaTeX pipeline (Overleaf sometimes prefers PDF); if PDF is needed, add a small conversion step and commit the PDFs (or a Makefile rule).

### 2026-02-10 (am) — Added reproducible SVG→PDF conversion hook (no system deps installed)

- Added `scripts/convert_figures_svg_to_pdf.sh` to convert `docs/paper/figures/*.svg` to `paper_figures/pdf/*.pdf` for LaTeX/Overleaf pipelines that prefer PDF.
- Updated `docs/paper/README.md` to document the SVG-as-source-of-truth convention and how to generate PDFs.
- Note: this machine currently lacks `rsvg-convert`/Inkscape, so the script is committed but PDFs are not generated here.

### 2026-02-10 (am) — Fixed experiment checklist to match current nlp16/GPU(4–7) policy

- Updated `docs/paper/HEARTBEAT_CHECKLIST.md` to align the Experiments lane with the current heartbeat prompt:
  - remote host: `nlp16`
  - repo: `/mnt/raid6/aa007878/galileo`
  - GPU policy: `CUDA_VISIBLE_DEVICES=4,5,6,7`

This prevents process drift where the checklist contradicts the heartbeat instructions.

### 2026-02-10 (am) — Reproducibility section updated to reflect artifact→figure pipeline

- Updated `docs/paper/PAPER_DRAFT_EN.md` Sec. 8 to explicitly describe:
  - required `paper_exports/` files (incl. metadata + runner metadata)
  - tracked artifacts under `docs/paper/artifacts/`
  - SVG source-of-truth figures under `docs/paper/figures/` + optional SVG→PDF conversion script

### 2026-02-10 (am) — Added figure inventory (figure→artifact mapping)

- Updated `docs/paper/README.md` with a concise figure inventory that maps each generated SVG in `docs/paper/figures/` to its tracked source artifact CSV under `docs/paper/artifacts/`.
- Goal: make it trivial (for us and for reviewers) to trace any plotted claim back to the committed artifact inputs.

### 2026-02-10 (am) — Paper drift fix: removed server-specific mention; clarified TOF↔turn-of-flip positioning

- Updated `docs/paper/PAPER_DRAFT_EN.md`:
  - Results preface no longer mentions a specific server name (avoid process drift in the draft).
  - Related Work positioning clarifies that our TOF is conceptually aligned with “turn-of-flip” metrics while retaining ground-truth + drift-control framing.

### 2026-02-10 (am) — Added SSH-sanity guardrail to prevent wasting experiment heartbeats

- Updated `docs/paper/HEARTBEAT_CHECKLIST.md` to require a quick `ssh nlp16 'hostname; whoami'` sanity check before selecting Experiments lane.
- If SSH is blocked, the checklist now explicitly advises switching the heartbeat to Writing/Development and logging the blocker, rather than spending the full 10 minutes on infra debugging.

### 2026-02-10 (am) — Centralized figure captions + provenance mapping

- Added `docs/paper/FIGURE_CAPTIONS.md` with paper-ready draft captions for each artifact-derived SVG figure, plus explicit figure→artifact provenance.
- Updated `docs/paper/README.md` to list `FIGURE_CAPTIONS.md` as a key file.

### 2026-02-10 (am) — Added nlp16 remote experiments runbook (heartbeat-friendly)

- Added `docs/paper/REMOTE_EXPERIMENTS_RUNBOOK.md` with copy-paste commands for:
  - SSH sanity check
  - mandatory tmux/GPU/results checks
  - log tailing for newest run
  - launch discipline and DM-update requirements
- Updated `docs/paper/README.md` to list the runbook.

### 2026-02-10 (am) — Updated submission checklist with figure pipeline requirements

- Updated `docs/paper/EMNLP_MAIN_SUBMISSION_CHECKLIST.md` (Sec. 7) to include explicit checklist items for the SVG→artifact provenance and optional PDF conversion, so final packaging doesn’t silently break figures.

### 2026-02-10 (am) — Added protocol overview figure (stdlib SVG)

- Added `scripts/make_protocol_figure_svg.py` (stdlib-only) to generate a clean, 1-column protocol diagram.
- Generated output: `docs/paper/figures/protocol_overview.svg`.
- Wired a short pointer into `docs/paper/PAPER_DRAFT_EN.md` (Sec. 3) so LaTeX conversion can include it.

### 2026-02-10 (am) — Protocol figure now has LaTeX include snippet + tightened caption

- Updated `docs/paper/PAPER_DRAFT_EN.md` (Sec. 3) to include a copy-pastable LaTeX `figure` block for `protocol_overview.svg` (label `fig:protocol`).
- Tightened the protocol figure caption to a single sentence in `docs/paper/FIGURE_CAPTIONS.md`.

### 2026-02-10 (am) — Added suggested LaTeX labels for all current figures

- Updated `docs/paper/FIGURE_CAPTIONS.md` to include suggested LaTeX `\label{...}` names for each figure (protocol + results figs), matching the labels used in `PAPER_DRAFT_EN.md`.
- Updated `docs/paper/PAPER_DRAFT_EN.md` to explicitly reference `Figure~\ref{fig:protocol}` for the protocol diagram.
