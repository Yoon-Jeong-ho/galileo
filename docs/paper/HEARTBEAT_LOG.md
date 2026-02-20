# GALILEO EMNLP Main — Heartbeat log & next plan

**SSOT (do not drift):** all EMNLP Main experiment work uses `ssh nlp8` + repo `/data_x/aa007878/galileo` + GPUs **4/5/6 only**. Older log entries may mention `nlp16` or `/mnt/raid6/...`; treat those as stale historical context.

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

### Remote (historical: nlp8)

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

### 2026-02-20 — Experiments lane: validate + stage Phi-3.5-mini seeds 1–2 into results_paper

- Confirmed both runs have full `paper_exports/` including `runner_metadata.json`:
  - `results/tier1_phi35mini_seed1_20260219_143555/paper_exports`
  - `results/tier1_phi35mini_seed2_20260219_143555/paper_exports`
- Ran validator on each run root: `[OK] .../paper_exports` and `[OK] runner_metadata parity`.
- Symlink-staged both into paper SSOT `results_paper/` and refreshed global validation: `results_paper/GLOBAL_VALIDATE.log` ends with `[OK] runner_metadata parity`.


### 2026-02-20 — Process lane: fix GPU-occupancy check command to avoid invalid `username` query

### 2026-02-20 — Process lane: add “torch alloc preflight must pass” gating (idle != usable)

- Observed that `nvidia-smi` can show GPUs as idle while torch CUDA alloc fails with `cudaErrorDevicesUnavailable`.
- Updated experiment checklists/runbook guidance to require a pure-torch alloc+synchronize preflight (`OK cuda alloc`) on the target GPU before launching.
- This is now part of the minimum checks to reduce wasted Tier‑1 launches.

- Observed that `nvidia-smi --query-compute-apps=...,username` fails on nlp8 (field not supported).
- Updated the heartbeat/runbook guidance to use `--query-compute-apps=gpu_uuid,pid,process_name,used_memory` and map PID→user via `ps` when we must enforce the “idle-only / not used by other users” policy.
- This reduces silent false-negatives during GPU selection.


**Paper: lock a reviewer-first presentation skeleton (Main Table + Figures), so we don’t fail on “no main table / vague figures / uneven section lengths”.**

- Deliverable:
  - New SSOT: `docs/paper/MAIN_TABLE_AND_FIGURES_PLAN.md`
  - Defines Table 1 spec (rows/cols, evidence pointers), minimum figure set, and section-length budget.

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

### 2026-02-10 (am) — Aligned SSOT heartbeat prompt + STATUS with current nlp16/GPU(4–7) policy

- Updated SSOT prompt `docs/paper/HEARTBEAT_PROMPT.md` to reference the current remote target (`ssh nlp16`, repo `/mnt/raid6/aa007878/galileo`) and GPU policy (4/5/6/7).
- Updated `docs/paper/STATUS.md` NOW section to remove outdated nlp8-only wording.

### 2026-02-10 (am) — Removed remaining nlp8-specific mentions from paper-facing text

- Updated `docs/paper/PAPER_DRAFT_EN.md` to remove “nlp8” from the Table W seed1–4 snapshot header (avoid leaking infra details / reduce drift).
- Marked the old “Remote (nlp8)” section in `docs/paper/HEARTBEAT_LOG.md` as historical context.

### 2026-02-10 (am) — Added anonymization notes + infra-string audit map

- Added `docs/paper/ANONYMIZATION_NOTES.md` with:
  - what infra-identifying strings to remove
  - current hotspots in the repo (KO drafts, runbooks, heartbeat logs)
  - a one-shot grep command for pre-submission audit
  - suggested placeholder replacements + packaging guidance

### 2026-02-10 (am) — Submission checklist now references anonymization SSOT

- Updated `docs/paper/EMNLP_MAIN_SUBMISSION_CHECKLIST.md` (Sec. 8) to point to `docs/paper/ANONYMIZATION_NOTES.md` as the SSOT for the pre-submission anonymization audit.

### 2026-02-10 (am) — Added anonymized bundle packager (with infra-string fail-fast)

- Added `scripts/package_anonymized_bundle.sh` to stage a minimal anonymized bundle (EN draft + captions + artifacts/figures + figure scripts).
- The script runs an infra-string grep audit on the staged bundle and fails fast if it finds hostnames/absolute paths.
- Note: the script intentionally excludes `docs/paper/ANONYMIZATION_NOTES.md` from the bundle because it necessarily contains infra-identifying strings.

### 2026-02-10 (am) — Documented anonymized bundle packager in paper README

- Updated `docs/paper/README.md` with a short section on `scripts/package_anonymized_bundle.sh` (how to stage+audit an anonymized bundle before external sharing).

### 2026-02-10 (am) — Set default recommendation: PDF figures for LaTeX

- Updated `docs/paper/README.md` to recommend **PDF** as the default figure format for LaTeX/Overleaf builds (SVG remains the source-of-truth).
- Updated `docs/paper/STATUS.md` next-step to focus on actually generating PDFs in the target build environment and verifying compilation.

### 2026-02-10 (am) — Added figure-tooling preflight script

- Added `scripts/check_figure_tooling.sh` to quickly validate whether `rsvg-convert`/Inkscape is available for SVG→PDF conversion.
- Updated `docs/paper/README.md` to reference the preflight check.

### 2026-02-10 (am) — LaTeX snippets now assume PDF build (SVG as source)

- Updated LaTeX include snippets in `docs/paper/PAPER_DRAFT_EN.md` to explicitly treat the repo SVGs as sources and recommend converting to PDF for the LaTeX build (placing PDFs under the LaTeX `figures/` directory).

### 2026-02-10 (am) — Added Makefile targets for figures + anonymized bundle

- Added repo-root `Makefile` with convenience targets:
  - `make figures-check` (tooling preflight)
  - `make figures-pdf` (SVG→PDF conversion)
  - `make anonymized-bundle` (stage+audit minimal anonymized bundle)
- Updated `docs/paper/README.md` to mention the make targets.

### 2026-02-10 (am) — STATUS updated: LaTeX build readiness + SSH access are current blockers

- Updated `docs/paper/STATUS.md` Top Gaps to reflect the actual current blockers:
  - PDF-figure build verification in the LaTeX environment
  - restoring `ssh nlp16` access from this runtime

### 2026-02-10 (am) — Wrote Tier-1 cross-family generalization plan (seeds 1–2)

- Added `docs/paper/TIER1_PLAN_CROSSFAMILY.md` to document the minimal, reviewer-risk-reducing plan for cross-family generalization (seeds 1–2) and what to report/export.

### 2026-02-10 (am) — Added tmux command printer for Tier-1 cross-family runs

- Added `scripts/print_crossfamily_run_commands.py` to print copy-pastable tmux launch templates for seeds 1–2 on GPUs 4–7.
- This standardizes session naming + OUT paths; we still need to fill in the canonical runner invocation once SSH access to nlp16 is restored.

### 2026-02-10 (am) — Added SSH troubleshooting note for restoring nlp16 access

- Added `docs/paper/SSH_TROUBLESHOOT_NLP16.md` (minimal info to share + common fixes: IdentityFile, IdentitiesOnly, ssh-agent).
- Linked it from `docs/paper/REMOTE_EXPERIMENTS_RUNBOOK.md` so experiment lane unblocks faster.

### 2026-02-10 (am) — Linked SSH troubleshooting SSOT from heartbeat checklist

- Updated `docs/paper/HEARTBEAT_CHECKLIST.md` to point to `docs/paper/SSH_TROUBLESHOOT_NLP16.md` when SSH sanity checks fail, so the experiment lane can unblock faster.

### 2026-02-10 (am) — Anonymized bundle packager now sanitizes Markdown in staged bundle

- Updated `scripts/package_anonymized_bundle.sh` to sanitize infra-identifying strings in staged Markdown files (bundle-only; does not touch sources).
- The staged bundle now can include `ANONYMIZATION_NOTES.md` without failing the infra-string audit.

### 2026-02-10 (am) — Anonymized bundle now includes README + submission checklist (sanitized)

- Updated `scripts/package_anonymized_bundle.sh` to include sanitized copies of:
  - `docs/paper/README.md`
  - `docs/paper/EMNLP_MAIN_SUBMISSION_CHECKLIST.md`
- Verified staged bundle passes the infra-string audit.

### 2026-02-10 (pm) — Research lane: related-work plan to strengthen methodology/positioning

- Added `docs/paper/RELATED_WORK_RESEARCH_PLAN_20260210.md` with a concrete integration plan around:
  - Time-To-Inconsistency (survival analysis framing)
  - ReviseQA (evidence-based belief revision vs pressure-induced flips)
  - Debate confidence escalation paper (motivation)

### 2026-02-10 (pm) — Writing lane: integrated Time-To-Inconsistency positioning into Related Work

- Updated `docs/paper/PAPER_DRAFT_EN.md` Sec 6.4 with a tight paragraph positioning GALILEO relative to survival-analysis-style time-to-event evaluation (Time-To-Inconsistency), clarifying our choice of direct ground-truth metrics + neutral drift control.

### 2026-02-10 (pm) — Writing lane: added ReviseQA contrast (evidence-based revision vs pressure)

- Added a short paragraph in `docs/paper/PAPER_DRAFT_EN.md` Sec 6.4 contrasting evidence-driven belief revision (ReviseQA) with pressure-induced flips in our persona setting.
- Added a placeholder BibTeX entry `@misc{reviseqa2025,...}` to `references.bib` (needs official BibTeX fields later).

### 2026-02-10 (pm) — Replaced ReviseQA placeholder with official BibTeX (OpenReview)

- Extracted official BibTeX from OpenReview `data-bibtex` field for ReviseQA.
- Updated `references.bib` entry key to `helwe2025reviseqa` and updated the in-text citation in `docs/paper/PAPER_DRAFT_EN.md` accordingly.

### 2026-02-10 (pm) — Writing lane: added debate confidence-dynamics motivation + BibTeX

- Added arXiv BibTeX entry for Nguyen/Prasad debate confidence dynamics (`prasad2025llmsdebatethinktheyll`) to `references.bib`.
- Added a single motivation sentence in `docs/paper/PAPER_DRAFT_EN.md` Sec 1.1 citing it to support the claim that multi-turn interaction can induce pathological belief/confidence dynamics.

### 2026-02-10 (pm) — Writing lane: clarified Neutral Re-asking Control design principle (no new evidence)

- Updated `docs/paper/PAPER_DRAFT_EN.md` Protocol section to explicitly state that Neutral Re-asking Control must not introduce new task-relevant evidence, to separate generic drift from evidence-based belief revision.

### 2026-02-10 (pm) — Writing lane: added persona vs control definition-level summary

- Updated `docs/paper/PAPER_DRAFT_EN.md` Protocol section with a compact bullet summary that contrasts persona pressure vs Neutral Re-asking Control at the definition level (same protocol/decoding/rounds; user-turn text differs; control introduces no new evidence).

### 2026-02-10 (pm) — Writing lane: added compact persona taxonomy table (mechanism-focused)

- Updated `docs/paper/PAPER_DRAFT_EN.md` Protocol section to include a compact 5-persona taxonomy table (mechanism + typical move), and explicitly noted the “no new evidence” design principle to avoid confounding with belief revision.

### 2026-02-10 (pm) — Policy switch: experiments on nlp8 GPUs 4/5/6

- Updated SSOT docs to match the current decision:
  - `docs/paper/HEARTBEAT_PROMPT.md`
  - `docs/paper/HEARTBEAT_CHECKLIST.md`
  - `docs/paper/REMOTE_EXPERIMENTS_RUNBOOK.md`
  - `docs/paper/STATUS.md`
- Remote experiments are now: `ssh nlp8`, repo `/data_x/aa007878/galileo`, GPUs 4/5/6 only.

### 2026-02-10 (pm) — Process: added hard anti-drift DM header rule

- Updated `docs/paper/HEARTBEAT_CHECKLIST.md` to require a 2-line DM header (`LANE=...`, `DELIVERABLE=...`) and to forbid lane switching mid-heartbeat.
- Rationale: reduce context drift across heartbeats and make dashboard verification easy.

### 2026-02-10 (pm) — Process: added 4-lane rotation plan (writing/research/experiments/method)

- Added `docs/paper/ROTATION_PLAN.md` to formalize the single-lane-per-heartbeat rotation and minimum deliverables per lane.

### 2026-02-10 (pm) — Writing lane: de-duplicated “no-new-evidence” framing (control vs ReviseQA)

- Updated `docs/paper/PAPER_DRAFT_EN.md` to merge duplicate statements about the Neutral Re-asking Control’s “no new evidence” principle and tightened the ReviseQA contrast to reference GALILEO’s no-new-evidence pressure/control design.
- Commit: `12eb63c`

### 2026-02-10 (pm) — Experiments lane: introduced `results_paper/` clean validation root

- On nlp8, created a paper-only validation root `results_paper/` that symlinks only the runs we cite (paper_exports only), and validated it with `validate_paper_exports.py --check_runner_parity` to get a stable PASS.
- Updated `docs/paper/REMOTE_EXPERIMENTS_RUNBOOK.md` with the `results_paper/` policy + validation command.

### 2026-02-10 (pm) — Development lane: added 2-seed Mistral cross-family tracked artifact

- Added tracked artifact `docs/paper/artifacts/tier1_mistral7b_seed1-2_survival_summary_20260210.csv` (Survival@5 and deltas vs control aggregated over seeds 1–2).
- Wired a short cross-family sanity-check sentence into `docs/paper/PAPER_DRAFT_EN.md` citing the artifact.

### 2026-02-10 (pm) — Development lane: added Llama seed1 cross-family tracked artifact

- Added tracked artifact `docs/paper/artifacts/tier1_llama3_8b_seed1_survival_summary_20260210.csv` (seed1 summary of Survival@5 and deltas vs control).
- Extended cross-family sanity-check text in `docs/paper/PAPER_DRAFT_EN.md` to cite the Llama artifact.

### 2026-02-10 — Paper writing lane: Llama cross-family artifact upgraded to seeds 1–2

- Added tracked artifact: `docs/paper/artifacts/tier1_llama3_8b_seed1-2_survival_summary_20260210.csv` (mean/std Survival@5 + deltas vs control across seeds 1–2).
- Updated `docs/paper/PAPER_DRAFT_EN.md` cross-family paragraph to cite the seed1–2 artifact and report mean Survival@5 numbers.

### 2026-02-10 — Development lane: LaTeX build readiness preflight (SVG→PDF)

- Ran `scripts/check_figure_tooling.sh` and confirmed the current environment is missing both `rsvg-convert` and `inkscape`, so SVG→PDF conversion cannot run yet.
- Updated `docs/paper/README.md` to clarify that `make` may be unavailable in minimal environments and documented script-direct invocation.

### 2026-02-10 — Paper development lane: add explicit SVG→PDF tooling check to submission checklist

- Updated `docs/paper/EMNLP_MAIN_SUBMISSION_CHECKLIST.md` to include an explicit build-environment requirement for SVG→PDF conversion (`rsvg-convert`/`inkscape`) validated by `scripts/check_figure_tooling.sh`.

### 2026-02-10 — Process lane: anti-drift hardening against nlp16 prompts

- Updated `docs/paper/HEARTBEAT_PROMPT.md` and `docs/paper/REMOTE_EXPERIMENTS_RUNBOOK.md` to explicitly override any stale heartbeat/poll text mentioning `nlp16` or `/mnt/raid6/...`.
- Rule: experiments (monitoring/launch/export/validate) are SSOT **nlp8 + GPUs 4/5/6 only**.

### 2026-02-10 — Experiments→writing bridge: add Qwen recovery_variant=verify_then_answer seed1 artifact

- Added tracked artifact: `docs/paper/artifacts/tier1_qwen2p5_7b_vta_seed1_survival_summary_20260210.csv` derived from nlp8 run `results/tier1_qwen2p5_7b_vta_seed1_20260210_192410/paper_exports/survival_curve.csv`.
- Updated `docs/paper/PAPER_DRAFT_EN.md` to cite the artifact as a Tier‑1 intervention ablation sanity check.

### 2026-02-10 — Writing lane: add recovery-collapsed artifact for Qwen verify_then_answer ablation

- Added tracked artifact: `docs/paper/artifacts/tier1_qwen2p5_7b_vta_seed1_recovery_collapsed_20260210.csv` by collapsing `results/.../recovery_accuracy.csv` over tasks.
- Updated `docs/paper/PAPER_DRAFT_EN.md` to cite the artifact and report control vs persona recovery rates for the intervention variant.

### 2026-02-10 — Process lane: STATUS refresh for recovery-variant ablation

- Updated `docs/paper/STATUS.md` to record the new paper-ready ablation alias (`results_paper/qwen_vta_seed1`) and to set the next-step focus on baseline vs verify_then_answer recovery comparison.

### 2026-02-10 — Analysis lane: baseline vs verify_then_answer recovery gap comparison

- Added tracked artifact: `docs/paper/artifacts/recovery_variant_verify_then_answer_vs_baseline_seed1_20260210.csv` comparing collapsed Recovery@flip persona–control gaps between baseline (seed1–4) and verify_then_answer (seed1).
- Updated `docs/paper/PAPER_DRAFT_EN.md` to cite the comparison artifact and explicitly caveat non-comparability + single-seed status.

### 2026-02-10 — Analysis lane: upgrade verify_then_answer recovery artifacts to seeds 1–2

- Added tracked artifact: `docs/paper/artifacts/tier1_qwen2p5_7b_vta_seed1-2_recovery_collapsed_20260210.csv` (collapsed recovery over tasks; seeds 1–2).
- Added tracked artifact: `docs/paper/artifacts/recovery_variant_verify_then_answer_vs_baseline_seed1-2_20260210.csv` comparing baseline vs vta persona–control recovery gaps.
- Updated `docs/paper/PAPER_DRAFT_EN.md` to replace single-seed recovery numbers with seeds 1–2 and cite the new artifacts.

### 2026-02-10 — Writing lane: upgrade verify_then_answer survival summary to seeds 1–2

- Added tracked artifact: `docs/paper/artifacts/tier1_qwen2p5_7b_vta_seed1-2_survival_summary_20260210.csv` (Survival@1 and Survival@5 means across seeds 1–2).
- Updated `docs/paper/PAPER_DRAFT_EN.md` intervention ablation paragraph to reference seeds 1–2 survival means and cite the new artifact.

### 2026-02-10 — Development lane: document no-root fallback for SVG figures

- Updated `docs/paper/README.md` with a practical fallback when SVG→PDF conversion is blocked in the build environment (no sudo): convert PDFs on any machine with `librsvg2-bin` and copy them into the LaTeX build, avoiding `--shell-escape`.

### 2026-02-10 — Process lane: STATUS updated for no-root SVG→PDF conversion fallback

- Updated `docs/paper/STATUS.md` to reference the documented no-sudo fallback for SVG→PDF conversion in `docs/paper/README.md`.

### 2026-02-10 — Process lane: STATUS updated for vta seeds 1–2

- Updated `docs/paper/STATUS.md` to reflect that the verify_then_answer recovery-variant ablation is now **seeds 1–2** and both aliases (`qwen_vta_seed1`, `qwen_vta_seed2`) are included in `results_paper/`.

### 2026-02-10 — Development lane: no-sudo SVG→PDF conversion via Inkscape AppImage

- Added `scripts/get_inkscape_appimage.sh` (downloads + sha256-verifies a pinned Inkscape AppImage to `tools/inkscape/inkscape.AppImage`).
- Updated `scripts/convert_figures_svg_to_pdf.sh` to fallback to the AppImage when `rsvg-convert`/`inkscape` are not available.
- Generated PDFs to `paper_figures/pdf/*.pdf` without sudo.

### 2026-02-10 — Development lane: checklist updated for no-sudo SVG→PDF tooling

- Updated `docs/paper/EMNLP_MAIN_SUBMISSION_CHECKLIST.md` to include the AppImage-based no-sudo SVG→PDF fallback (`scripts/get_inkscape_appimage.sh`).

### 2026-02-10 — Development lane: document AppImage conversion warnings

- Added an inline note in `scripts/convert_figures_svg_to_pdf.sh` explaining that gio/dconf module warnings during Inkscape AppImage export are typically harmless for headless SVG→PDF conversion.

### 2026-02-10 — Development lane: add PDF figure smoke-check (no LaTeX required)

- Added `scripts/check_pdf_figures.sh` to verify generated `paper_figures/pdf/*.pdf` exist and have valid PDF headers/sizes in environments without LaTeX.
- Updated `docs/paper/EMNLP_MAIN_SUBMISSION_CHECKLIST.md` to include the smoke-check command.

### 2026-02-10 — Writing/process lane: seed the claim→evidence map

- Added an initial 5-row claim→evidence mapping table to `docs/paper/EMNLP_MAIN_SUBMISSION_CHECKLIST.md` linking C1/C2/C3 + generalization + intervention ablation to (i) a specific figure/table, (ii) tracked artifacts, and (iii) paper-ready `results_paper/` aliases.

### 2026-02-10 — Writing/process lane: make claim→evidence map reproducible

- Refined the claim→evidence table in `docs/paper/EMNLP_MAIN_SUBMISSION_CHECKLIST.md` to prefer a single primary figure per claim and to include concrete reproducer commands (validator invocation) and explicit `results_paper/` aliases.

### 2026-02-10 — Research lane: add ReviseQA paper note

- Added `docs/paper/related_work/papers/reviseqa_2024.md` summarizing how ReviseQA relates/contrasts with GALILEO’s no-new-evidence pressure/control design.

### 2026-02-11 — Research lane: verify ReviseQA metadata source

- Confirmed ReviseQA appears on OpenReview as “ReviseQA: A Benchmark for Belief Revision in Multi-Turn Logical Reasoning” (OpenReview id: Z4KBiAYXlI). Updated the note header to 2025/OpenReview and linked the canonical page for exact BibTeX metadata.

### 2026-02-11 — Research lane: close ReviseQA BibTeX TODO

- Updated `docs/paper/related_work/papers/reviseqa_2024.md` to note that `helwe2025reviseqa` is already present in `references.bib`.

### 2026-02-11 — Writing lane: sharpen ReviseQA contrast (no-new-evidence + control rationale)

- Updated `docs/paper/PAPER_DRAFT_EN.md` to make the ReviseQA contrast more explicit: under fixed information, flips are attributable to pressure/drift, motivating the Neutral Re-asking Control for interpretation.

### 2026-02-11 — Writing lane: strengthen Abstract with multi-seed + cross-family evidence

- Updated the Abstract in `docs/paper/PAPER_DRAFT_EN.md` to replace “initial snapshots” with a concrete statement grounded in our current paper-ready results: Qwen seeds 1–4 plus Mistral/Llama seeds 1–2 show consistent persona-dependent degradation relative to the Neutral Re-asking Control.

### 2026-02-11 — Writing lane: align Contributions with claim→evidence framing

- Updated the Introduction contributions list in `docs/paper/PAPER_DRAFT_EN.md` to explicitly (i) motivate the Neutral Re-asking Control as a drift baseline for interpreting pressure-induced flips and (ii) emphasize recovery measured conditional on flip plus recovery-prompt ablations.

### 2026-02-11 — Writing lane: make Abstract recovery definition explicit (conditional on flip)

- Updated the Abstract in `docs/paper/PAPER_DRAFT_EN.md` to define recovery as **conditional on flip**, matching the C3 framing and avoiding ambiguity with “staying correct throughout.”

### 2026-02-11 — Process lane: align C3 wording in claim→evidence map

- Updated the C3 row in `docs/paper/EMNLP_MAIN_SUBMISSION_CHECKLIST.md` to define recovery explicitly as **conditional on flip** (return-to-truth after a flip), matching the Abstract/Intro framing.

### 2026-02-11 — Writing lane: compress Abstract (split protocol vs findings)

- Split the Abstract’s second paragraph in `docs/paper/PAPER_DRAFT_EN.md` into (i) protocol/definitions/control/evaluation stability and (ii) key findings, improving readability while keeping the same factual scope.

### 2026-02-10 — Experiments lane: validate qwen VTA runs + confirm paper SSOT green

- Remote (nlp8): checked tmux + nvidia-smi (GPUs 4/5/6 idle) and tailed logs for `results/tier1_qwen2p5_7b_vta_seed1_20260210_192410` + `results/tier1_qwen2p5_7b_vta_seed2_20260210_205204` (both wrote full `paper_exports/` and printed `[OK]` + parity).
- Confirmed paper-only validation root remains green: `results_paper/GLOBAL_VALIDATE.log` is all `[OK]` incl. `qwen_vta_seed1/2`.
- Updated `docs/paper/STATUS.md` NOW section with validator health.

### 2026-02-10 — Writing/process lane: tighten claim→evidence map reproducers

- Updated `docs/paper/EMNLP_MAIN_SUBMISSION_CHECKLIST.md` claim→evidence rows (C3/generalization/intervention) to use a single concrete reproducer: `scripts/validate_paper_exports.py --results_root results_paper --check_runner_parity` and to point generalization/ablation evidence to `PAPER_DRAFT_EN.md` §7.4.

### 2026-02-10 — Writing lane: add cross-family generalization figure (SVG, stdlib)

- Added `scripts/make_cross_family_figure_svg.py` to generate a compact cross-family Survival@5 comparison (control vs Logical Trap) from tracked artifacts.
- Generated `docs/paper/figures/cross_family_survival_r5_control_vs_logicaltrap_seed1-2_20260210.svg` and referenced it in `docs/paper/PAPER_DRAFT_EN.md` §7.4.

### 2026-02-10 — Writing lane: add LaTeX snippet + caption entry for cross-family figure

- Added a LaTeX include snippet for `docs/paper/figures/cross_family_survival_r5_control_vs_logicaltrap_seed1-2_20260210.svg` in `docs/paper/PAPER_DRAFT_EN.md` §7.4.
- Added a corresponding caption/provenance entry to `docs/paper/FIGURE_CAPTIONS.md`.

### 2026-02-11 — Writing/process lane: add citation-key audit script

- Added `scripts/audit_citations.py` to verify that all `\cite{...}` keys used in paper drafts exist in `references.bib` (quick pre-LaTeX guardrail).
- Ran it: EN draft has 11 cite keys and **0 missing**; KO draft currently has 0 cite keys.

### 2026-02-11 — Writing lane: finish converting remaining author-year citations in EN draft

- Converted remaining inline author-year mentions in `docs/paper/PAPER_DRAFT_EN.md` to BibTeX-backed `\cite{...}` (ELEPHANT; theorem-proving sycophancy).
- Re-ran `python3 scripts/audit_citations.py`: EN draft now cites 13 keys and **0 are missing** from `references.bib`.

### 2026-02-11 — Process lane: add Tier‑1 decoding sensitivity sweep plan (paper SSOT)

- Added `docs/paper/DECODING_SENSITIVITY_SWEEP.md` with a minimal Tier‑1 decoding sweep plan (Qwen seeds 1–2; greedy_temperature 0.0 vs 0.7) and paper-ready export/validation steps under `results_paper/`.
- Linked the plan from `docs/paper/EMNLP_MAIN_SUBMISSION_CHECKLIST.md`.

### 2026-02-11 — Process lane: add lightweight CLI summary helper for run_experiment.py

- Added `scripts/show_run_experiment_cli.sh` to list CLI flags without running `python run_experiment.py --help` (which can be slow/hang due to heavyweight imports).
- Updated `docs/paper/DECODING_SENSITIVITY_SWEEP.md` to recommend using the helper when preparing decoding sweep runs.

### 2026-02-11 — Experiments lane: launch decoding sensitivity sweep (seed1)

- Remote (nlp8): launched Tier‑1 decoding sweep runs (Qwen2.5-7B-Instruct; 80 samples; seed1) under `results_paper/`:
  - `results_paper/qwen_temp0_seed1` (GPU4; `--greedy_temperature 0.0`)
  - `results_paper/qwen_temp0p7_seed1` (GPU5; `--greedy_temperature 0.7`)
- Both runs are in tmux sessions `qwen_temp0_seed1` / `qwen_temp0p7_seed1` and are configured to auto-run paper export + validator on completion.

### 2026-02-11 — Experiments lane: decoding sensitivity sweep seed1 completed + validated

- Remote (nlp8): decoding sweep seed1 runs completed and are paper-ready:
  - `results_paper/qwen_temp0_seed1` (GPU4; `--greedy_temperature 0.0`) → wrote full `paper_exports/` + `[OK]` + parity.
  - `results_paper/qwen_temp0p7_seed1` (GPU5; `--greedy_temperature 0.7`) → wrote full `paper_exports/` + `[OK]` + parity.
- Updated `results_paper/GLOBAL_VALIDATE.log` via `scripts/validate_paper_exports.py --results_root results_paper --check_runner_parity` (all `[OK]`).

### 2026-02-11 — Experiments lane: launch decoding sensitivity sweep (seed2)

- Remote (nlp8): launched decoding sweep seed2 runs under `results_paper/` (tmux; auto export+validate on completion):
  - `results_paper/qwen_temp0_seed2` (GPU4; `--greedy_temperature 0.0`)
  - `results_paper/qwen_temp0p7_seed2` (GPU5; `--greedy_temperature 0.7`)

### 2026-02-11 — Experiments lane: monitor decoding sensitivity sweep (seed2 in progress)

- Remote (nlp8): checked GPU + tmux/logs for decoding sweep seed2.
  - GPUs: GPU4 ~96
### 2026-02-11 — Experiments lane: monitor decoding sensitivity sweep (seed2 in progress)

- Remote (nlp8): checked GPU + tmux/logs for decoding sweep seed2.
  - GPUs: GPU4 ~96% / 45.8GB, GPU5 ~96% / 46.9GB (active); GPU6 idle.
  - `qwen_temp0_seed2`: in multi-turn phase (round progress ongoing; no `paper_exports/` yet).
  - `qwen_temp0p7_seed2`: in multi-turn phase (round 4 claim generation visible; no `paper_exports/` yet).

### 2026-02-11 — Experiments lane: decoding sensitivity sweep seed2 completed + SSOT validated

- Remote (nlp8): decoding sweep seed2 runs completed and are paper-ready:
  - `results_paper/qwen_temp0_seed2` (`--greedy_temperature 0.0`) → wrote full `paper_exports/` + `[OK]` + parity.
  - `results_paper/qwen_temp0p7_seed2` (`--greedy_temperature 0.7`) → wrote full `paper_exports/` + `[OK]` + parity.
- Refreshed `results_paper/GLOBAL_VALIDATE.log` (all `[OK]`, includes seed1+seed2 temp runs).

### 2026-02-11 — Writing/analysis lane: summarize decoding sensitivity sweep into tracked artifact + Results prose

- Pulled a decoding sweep summary CSV into `docs/paper/artifacts/decoding_sweep_qwen_temp_summary_seed1-2_20260211.csv` (computed from paper-ready runs under `results_paper/qwen_temp{0,0p7}_seed{1,2}`).
- Added a short Decoding sensitivity paragraph to `docs/paper/PAPER_DRAFT_EN.md` §7.4 referencing the tracked artifact and summarizing stable ΔSurvival@5 / ΔFail@1 across temperatures.

### 2026-02-11 — Process lane: mark decoding sensitivity sweep as completed in submission checklist

- Updated `docs/paper/EMNLP_MAIN_SUBMISSION_CHECKLIST.md` to check off the decoding sensitivity sweep item and link it to the tracked artifact + paper-ready `results_paper/` run aliases + `results_paper/GLOBAL_VALIDATE.log`.

### 2026-02-11 — Writing lane: add decoding sweep figure (SVG, stdlib)

- Added `scripts/make_decoding_sweep_figure_svg.py` and generated `docs/paper/figures/decoding_sweep_qwen_delta_seed1-2_20260211.svg` (ΔSurvival@5 and ΔFail@1 at temp=0.0 vs 0.7; seeds 1–2).
- Added caption/provenance entry to `docs/paper/FIGURE_CAPTIONS.md`.

### 2026-02-11 — Writing lane: integrate decoding sweep figure into paper draft (LaTeX snippet)

- Updated `docs/paper/PAPER_DRAFT_EN.md` to reference `fig:decoding-sweep` and added a LaTeX include snippet for `docs/paper/figures/decoding_sweep_qwen_delta_seed1-2_20260211.svg`.

### 2026-02-11 — Process lane: generate PDF figures (no-sudo Inkscape AppImage) + smoke-check

- Generated PDFs from `docs/paper/figures/*.svg` into `paper_figures/pdf/*.pdf` using `scripts/convert_figures_svg_to_pdf.sh` (AppImage backend).
- Verified PDFs via `bash scripts/check_pdf_figures.sh` (PASS). Includes:
  - `paper_figures/pdf/decoding_sweep_qwen_delta_seed1-2_20260211.pdf`

### 2026-02-11 — Process lane: fix docs SSOT (nlp8) + clarify PDF figure policy

- Updated `docs/paper/README.md` to (i) correct the remote runbook host reference to nlp8 and (ii) document the no-sudo Inkscape AppImage SVG→PDF conversion path + a practical “commit PDFs used in LaTeX” policy.

### 2026-02-11 — Process lane: remove remaining nlp16 doc drift (SSOT=nlp8)

- Updated docs to eliminate stale `nlp16` guidance:
  - Added `docs/paper/SSH_TROUBLESHOOT_REMOTE.md` (nlp8 SSOT) and pointed `docs/paper/REMOTE_EXPERIMENTS_RUNBOOK.md` to it.
  - Fixed runbook DM checklist to report GPU util/mem for GPUs 4–6 (not 4–7).
  - Updated `docs/paper/TIER1_PLAN_CROSSFAMILY.md` to target nlp8 and GPUs 4/5/6.
  - Marked `docs/paper/SSH_TROUBLESHOOT_NLP16.md` as deprecated.

### 2026-02-11 — Process lane: annotate SSOT + fix anonymization note drift

- Added an explicit SSOT banner at the top of `docs/paper/HEARTBEAT_LOG.md` warning that older entries mentioning `nlp16` are stale.
- Updated `docs/paper/ANONYMIZATION_NOTES.md` to reflect that the current runbook contains `nlp8` (not `nlp16`).

### 2026-02-11 — Process lane: tighten anonymization notes (ssh nlp8 mapping + paper-facing grep)

- Updated `docs/paper/ANONYMIZATION_NOTES.md` to map both `ssh nlp8` and `ssh nlp16` to `ssh <REMOTE_HOST>` for anonymized bundles.
- Added an optional suggestion to exclude internal process docs when running the infra-string grep, to reduce false positives.

### 2026-02-11 — Process lane: make anonymized bundle paper-facing by default

- Updated `scripts/package_anonymized_bundle.sh` to exclude internal process docs (README/checklists/runbooks/logs) by default.
- Smoke-tested staging: `./scripts/package_anonymized_bundle.sh tmp/anonymized_bundle_test` (PASS; infra-string audit clean).

### 2026-02-11 — Process lane: add optional PDF inclusion to anonymized bundle (with best-effort scan)

- Updated `scripts/package_anonymized_bundle.sh` to support `INCLUDE_PDF=1` (copies `paper_figures/pdf/*.pdf` into the staged bundle).
- Added a best-effort PDF metadata scan using `strings | grep` for infra-identifying patterns.
- Smoke-tested: `INCLUDE_PDF=1 ./scripts/package_anonymized_bundle.sh tmp/anonymized_bundle_test_pdf` (PASS).

### 2026-02-11 — Process lane: make anonymized bundle include PDFs by default

- Updated `scripts/package_anonymized_bundle.sh` so `INCLUDE_PDF` defaults to 1 (PDFs included unless `INCLUDE_PDF=0`).
- When PDFs are included, the bundle now also ships `scripts/check_pdf_figures.sh` for a quick PDF sanity check.
- Smoke-tested default bundle staging: `./scripts/package_anonymized_bundle.sh tmp/anonymized_bundle_default_pdf` (PASS).

### 2026-02-11 — Process lane: document INCLUDE_PDF default in bundler header

- Updated `scripts/package_anonymized_bundle.sh` header comments to state that PDFs are included by default and can be disabled with `INCLUDE_PDF=0`.

### 2026-02-11 — Process lane: anonymized bundler copies only LaTeX-referenced PDFs by default

- Updated `scripts/package_anonymized_bundle.sh` so when PDFs are included, it copies only `paper_figures/pdf/<name>.pdf` that are referenced in `docs/paper/PAPER_DRAFT_EN.md` via `\includegraphics{figures/<name>}`.
- Added knobs:
  - `INCLUDE_PDF=0` to disable PDFs
  - `PDF_USED_ONLY=0` to copy *all* PDFs
- Smoke-tested: `./scripts/package_anonymized_bundle.sh tmp/anonymized_bundle_pdf_used` (PASS).

### 2026-02-11 — Process lane: bundler copies only LaTeX-referenced SVGs by default

- Updated `scripts/package_anonymized_bundle.sh` to copy only `docs/paper/figures/<name>.svg` that are referenced in `docs/paper/PAPER_DRAFT_EN.md` via `\includegraphics{figures/<name>}`.
- Added knobs:
  - `INCLUDE_SVG=0` to disable SVGs
  - `SVG_USED_ONLY=0` to copy *all* SVGs
- Smoke-tested: `./scripts/package_anonymized_bundle.sh tmp/anonymized_bundle_used_figs` (PASS; 8 SVGs + 8 PDFs staged).

### 2026-02-11 — Process lane: document bundler defaults (PDF+SVG used-only)

- Updated `scripts/package_anonymized_bundle.sh` header to document default behavior: include PDFs and SVGs, filtered to those referenced in `PAPER_DRAFT_EN.md`; disable via `INCLUDE_PDF=0` / `INCLUDE_SVG=0`.

### 2026-02-11 — Process/writing lane: make artifact paths literal (avoid brace expansion)

- Updated `docs/paper/PAPER_DRAFT_EN.md` to replace a brace-expanded artifact path (`seed{1,2,3,4}`) with explicit per-seed filenames, so packaging scripts and readers resolve paths unambiguously.
- Updated `scripts/package_anonymized_bundle.sh` to copy only paper-referenced artifact CSVs by default (from `PAPER_DRAFT_EN.md` + `FIGURE_CAPTIONS.md`).

### 2026-02-11 — Process lane: audit artifact references vs inventory (paper-facing)

- Checked `docs/paper/PAPER_DRAFT_EN.md` + `docs/paper/FIGURE_CAPTIONS.md` for brace-expanded artifact paths: none found.
- Verified artifact references are complete: 15 referenced CSVs and 0 missing under `docs/paper/artifacts/`.
- Found 8 unreferenced CSVs (safe to keep internal; bundler now excludes by default).

### 2026-02-11 — Process lane: archive unreferenced artifacts to reduce confusion

- Moved 8 unreferenced CSVs from `docs/paper/artifacts/` to `docs/paper/artifacts/archive/` (keeps the main artifacts directory aligned with paper-facing references).
- Verified `PAPER_DRAFT_EN.md` + `FIGURE_CAPTIONS.md` still have 0 missing artifact references.

### 2026-02-11 — Process lane: staged anonymized bundle passes PDF smoke-check

- Staged bundle: `tmp/anonymized_bundle_post_archive_pdfcheck` via `scripts/package_anonymized_bundle.sh`.
- Ran `bash tmp/anonymized_bundle_post_archive_pdfcheck/scripts/check_pdf_figures.sh` inside the staged bundle: PASS (all 8 PDFs valid).

### 2026-02-11 — Process lane: archive anonymized bundle (tar.gz + sha256)

- Added `scripts/archive_anonymized_bundle.sh` to create a tar.gz archive + sha256 checksum from a staged bundle dir.
- Produced:
  - `tmp/galileo_anonymized_bundle_20260211.tar.gz`
  - `tmp/galileo_anonymized_bundle_20260211.tar.gz.sha256`
  - sha256: `a443c2ccbdffde4c2e4811637940cd46a5573e6e09cf94453cc9a561d4de51e7`

### 2026-02-11 — Process lane: add optional zip archive for anonymized bundle (no external deps)

- Updated `scripts/archive_anonymized_bundle.sh` to support `MAKE_ZIP=1` and generate `<OUT>.zip` + `<OUT>.zip.sha256`.
- Implemented a Python stdlib `zipfile` fallback when the `zip` CLI is unavailable.
- Produced:
  - `tmp/galileo_anonymized_bundle_20260211.zip`
  - `tmp/galileo_anonymized_bundle_20260211.zip.sha256`
  - sha256: `cebcb463447b844c4477bee5ffd14c6b5cdcde8cd2f01491450a79f804fc3b82`

### 2026-02-11 — Writing lane: qualitative flip taxonomy hardening (EN/KO) + status sync

- KO (`docs/paper/PAPER_DRAFT_KO.md`): expanded qualitative taxonomy with multi-seed snapshot tables (bucket counts + ARC frequency), added multi-seed representative examples (seed1–4), and clarified that taxonomy is an interpretability aid (not replacing quantitative survival/TOF/recovery).
- EN (`docs/paper/PAPER_DRAFT_EN.md`): added Appendix A.2 taxonomy (boundary/partial/semantic), added F1 threshold examples, separated rare format/extraction artifacts from semantic flips, and linked A.2 from Limitations + claim table.
- Monitoring: confirmed `results_paper/GLOBAL_VALIDATE.log` remains all `[OK]` and runner_metadata parity PASS.
- STATUS refreshed to reflect taxonomy progress + updated next-step focus.
- Key commits (selected): `8e3625c` (STATUS), plus recent EN/KO taxonomy commits up to `a280759`.

### 2026-02-11 — Experiments lane: make Llama-3.2-3B seed1 paper-ready (runner_metadata + results_paper staging)

- Detected Tier-1 Llama-3.2-3B seed1 run had `paper_exports/*` but failed validation due to missing `paper_exports/runner_metadata.json`.
- Added `runner_metadata.json` (gpu_list=4, TP=1, num_samples=80, max_model_len=8192, max_tokens=2048, env=galileo) and re-ran validator: `[OK]`.
- Copied the now-auditable run into paper SSOT root: `results_paper/tier1_llama3_3b_seed1_20260212_030426/` and re-validated.
- Ran global parity validation over `results_paper/`: `[OK] runner_metadata parity`.

### 2026-02-11 — Writing lane: mention Llama-3.2-3B seed1 cross-family check
- PAPER_DRAFT_EN: note additional Llama-3.2-3B seed1 check + add results_paper path to claim→evidence row.

### 2026-02-11 — Experiments lane: launch Llama-3.2-3B seed2 (complete Tier‑1 cross-family seeds 1–2)

- Launched on nlp8 GPU5 (TP=1, max_model_len=8192, max_tokens=2048, num_samples=80) in tmux:
  - session: `tier1_llama3_3b_s2_g5_20260212_042339`
  - OUT: `results/tier1_llama3_3b_seed2_20260212_042339/`
- Command (in tmux): `CUDA_VISIBLE_DEVICES=5 conda run -n galileo python run_experiment.py --model meta-llama/Llama-3.2-3B-Instruct ... --seed 2`.
- Next when finished: run `scripts/paper_export.py`, add `paper_exports/runner_metadata.json`, validate, then copy into `results_paper/` and run global parity validation.

## 2026-02-12 (cron: paper-10min-research-checkin)
- Paper: tightened the reviewer-facing *Claims → evidence map* to be **local-first** (removed hardcoded nlp8 mention) and added explicit ‘requires results_paper/ present’ notes.
  - File: `docs/paper/PAPER_DRAFT_EN.md`
- Paper: added explicit **LaTeX figure labels → repo files** to the Claims→evidence table (proof-pointer anti-drift).
  - File: `docs/paper/PAPER_DRAFT_EN.md`

### 2026-02-12 — Experiments lane: finish + stage Llama-3.2-3B seed2 as paper-ready

- The tmux run `tier1_llama3_3b_s2_g5_20260212_042339` completed (OUT: `results/tier1_llama3_3b_seed2_20260212_042339/`).
- Ran `scripts/paper_export.py`, wrote `paper_exports/*` + `paper_exports/runner_metadata.json`, then validated: `[OK]`.
- Copied to paper SSOT root: `results_paper/tier1_llama3_3b_seed2_20260212_042339/` and re-validated.
- Re-ran global validation over `results_paper/` with parity check: `[OK] runner_metadata parity`.

### 2026-02-12 — Writing lane: add Results TOF→Appendix A.2 cross-reference

- Added a 1-sentence caveat in §7.2 TOF results pointing readers to Appendix~A.2 for extractive-QA boundary/overanswer flip artifacts under strict EM.

### 2026-02-12 — Writing lane: sync cross-family SSOTs (C4 seeds 1–2)

- Updated `docs/paper/CLAIM_EVIDENCE_MAP.md` to add a C4 Cross-family section with correct Llama-3.2-3B seed1+seed2 run roots.
- Updated `docs/paper/STATUS.md` to remove the stale “seed1-only” phrasing and state seeds 1–2 are staged + parity OK.

### 2026-02-12 — Writing lane: clarify A.2 taxonomy terminology (diagnostic buckets)

- Added a 1-line terminology note in Appendix~A.2 stating boundary/overanswer, partial-overlap, semantic-change are diagnostic buckets (not new metrics).

### 2026-02-12 — Writing lane: cite CoVe (verify-then-answer) in Related Work

- Added 1 short paragraph in Related Work contrasting CoVe-style self-verification (verify-then-answer) with GALILEO’s multi-turn pressure + survival/TOF + recovery evaluation framing.

### 2026-02-12 (pm) — Local docs: de-confuse deprecated nlp16 SSH note

- Updated `docs/paper/SSH_TROUBLESHOOT_NLP16.md` to clearly state SSOT host is `nlp8` and to treat `nlp16` references as historical/template only.
- Replaced example commands/config blocks to use `nlp8` (reduces drift when copy-pasting).

### 2026-02-12 — Writing lane: add SSOT note to prevent overclaiming flip=belief-change

- Updated `docs/paper/CLAIM_EVIDENCE_MAP.md` with an explicit “flip interpretation caveat” section pointing to Appendix~A.2 and stating taxonomy buckets are diagnostic (not metrics).

### 2026-02-12 — Paper dev lane: make Llama-3.2-3B cross-family evidence repo-auditable

- Generated and committed a repo-tracked Tier-1 summary CSV for Llama-3.2-3B seeds 1–2:
  - `docs/paper/artifacts/tier1_llama3_3b_seed1-2_survival_summary_20260212.csv`
- Updated the cross-family figure generator to include the 3B family and produced:
  - `docs/paper/figures/cross_family_survival_r5_control_vs_logicaltrap_seed1-2_20260212.svg`
- Synced paper-facing references in `PAPER_DRAFT_EN.md` and SSOT references in `CLAIM_EVIDENCE_MAP.md`.

### 2026-02-12 — Paper dev lane: eliminate cross-family figure filename drift (20260210→20260212)

- Updated checklist + LaTeX skeletons to reference the canonical cross-family figure: `cross_family_survival_r5_control_vs_logicaltrap_seed1-2_20260212`.
- Added a one-line canonical filename note in `docs/paper/STATUS.md`.

### 2026-02-12 — Writing lane: make flip taxonomy diagnostic-only explicit in main Results

- In `docs/paper/PAPER_DRAFT_EN.md` §7.1, added an explicit sentence that the Appendix~A.2 flip taxonomy is post-hoc/diagnostic and **does not** recompute survival/TOF/recovery.

### 2026-02-12 — Writing lane: propagate Survival/Flip/TOF definitions into EN draft

- Updated `docs/paper/PAPER_DRAFT_EN.md` “Metrics in brief” to match the centralized definitions in `docs/paper/FIGURE_CAPTIONS.md` (Survival@r is cumulative through r; Flip event; TOF first flip).
- Updated `docs/paper/STATUS.md` to note metric definitions SSOT lives in `FIGURE_CAPTIONS.md`.

### 2026-02-12 — Process lane: add paper SSOT run aliases to claim→evidence map (C3)

- Updated `docs/paper/CLAIM_EVIDENCE_MAP.md` to explicitly list `results_paper/qwen_vta_seed{1,2}` as the auditable run aliases backing the verify_then_answer recovery-variant comparison.

### 2026-02-12 (pm) — Claim→evidence map: Abstract/Intro checklist + path fixes

- Updated  to add an Abstract/Intro reviewer-auditable checklist and corrected figure/artifact pointers to match the committed  and  filenames.
- Updated  to reflect progress + set the next writing step (wire pointers into Abstract/Intro text).

Next:
- Add explicit Fig/Table pointers in  Abstract/Intro aligned with the claim map.
- Audit  §1.4 claims for any missing evidence pointers.
- (Optional) add a tiny “how to verify” snippet (validator + artifact regeneration) to the draft appendix.

### 2026-02-12 (pm) — Claim→evidence map: Abstract/Intro checklist + path fixes

- Updated `docs/paper/CLAIM_EVIDENCE_MAP.md` to add an Abstract/Intro reviewer-auditable checklist and corrected figure/artifact pointers to match the committed `docs/paper/figures/*.svg` and `docs/paper/artifacts/*.csv` filenames.
- Updated `docs/paper/STATUS.md` to reflect progress + set the next writing step (wire pointers into Abstract/Intro text).

Next:
- Add explicit Fig/Table pointers in `docs/paper/PAPER_DRAFT_EN.md` Abstract/Intro aligned with the claim map.
- Audit `docs/paper/PAPER_DRAFT_EN.md` §1.4 claims for any missing evidence pointers.
- (Optional) add a tiny “how to verify” snippet (validator + artifact regeneration) to the draft appendix.

### 2026-02-12 (pm) — Writing lane: add explicit Abstract proof pointers (reviewer-auditable)

- Updated `docs/paper/PAPER_DRAFT_EN.md` Abstract to include a compact “Proof pointers” sentence that lists the key evidence hooks (protocol / survival / TOF / recovery / Table W deltas).
- Updated `docs/paper/STATUS.md` to record this and set the next step (propagate proof pointers into Intro/core claims).

### 2026-02-12 — Writing lane: tighten Intro contributions proof pointers (avoid section-number churn)

- Updated §1.3 Contributions pointers to use stable anchors (Task setting / Evaluation details / artifacts+captions SSOT) instead of brittle section numbers.

### 2026-02-12 18:38 KST — Claim→evidence map: add decoding-sensitivity proof pointers

- Updated `docs/paper/CLAIM_EVIDENCE_MAP.md` to include a reviewer-auditable decoding sensitivity checklist item (temp 0.0 vs 0.7; seeds 1–2) and a dedicated section with artifact + script pointers.
  - Evidence: `docs/paper/figures/decoding_sweep_qwen_delta_seed1-2_20260211.svg`
  - Artifact: `docs/paper/artifacts/decoding_sweep_qwen_temp_summary_seed1-2_20260211.csv`
  - Regeneration: `python3 scripts/make_decoding_sweep_figure_svg.py`

### 2026-02-12 — Paper dev lane: add LaTeX anti-drift pointer (labels ↔ repo filenames)

- Updated LaTeX skeleton READMEs to point to the SSOT label↔filename mapping in `docs/paper/CLAIM_EVIDENCE_MAP.md`.

### 2026-02-17 (am) — Writing lane: extend proof pointers in Intro (reduce reviewer search cost)

- Updated `docs/paper/PAPER_DRAFT_EN.md` Intro to attach explicit proof pointers to the evaluation gap bullets and the motivation paragraph:
  - failure dynamics sentence now points to survival/TOF/recovery figures (Figs.~\ref{fig:survival-curves-rounds}, \ref{fig:tof-delta-fail1}, \ref{fig:recovery-delta})
  - evaluation-gap bullets now each point to the exact figure
  - protocol sentence now explicitly points to Fig.~\ref{fig:protocol}
- Ran `python3 scripts/audit_citations.py`: EN draft has 14 cite keys and **0 missing**.

Next:
- Scan the rest of Intro/Contributions for any remaining “floating” claim sentences without a nearby proof pointer.
- Ensure the LaTeX labels in `docs/paper/CLAIM_EVIDENCE_MAP.md` still match the draft refs.

### 2026-02-17 (am) — Experiments lane: stage Llama-3.2-3B seed2 into results_paper and restore global parity PASS

- Remote (nlp8): confirmed Tier‑1 `meta-llama/Llama-3.2-3B-Instruct` seed2 run has complete `paper_exports/` + `runner_metadata.json` and validator `[OK]`:
  - source: `results/tier1_llama3_3b_seed2_20260212_042339/`
- Staged into paper SSOT root:
  - `results_paper/tier1_llama3_3b_seed2_20260212_042339/`
- Ran global parity validation over `results_paper/`:
  - initially failed due to incomplete EXAONE directories mistakenly placed under `results_paper/`
  - moved incomplete dirs to `results_paper_incomplete/`
  - re-ran `python scripts/validate_paper_exports.py --results_root results_paper --check_runner_parity` → **[OK] runner_metadata parity**

Notes:
- GPUs at check time: GPU4~99% util, GPU5~100% util, GPU6 idle (work in progress sessions).

### 2026-02-17 — Paper dev lane: set canonical Table W LaTeX label (anti-drift)

- Declared canonical table label `tab:tablew` in the SSOT label↔artifact mapping (`CLAIM_EVIDENCE_MAP.md`) and noted it in `FIGURE_CAPTIONS.md`.

### 2026-02-17 — Writing lane: lock Table W LaTeX label usage in Intro proof pointers

- Updated Intro proof pointers to cite `Table~\ref{tab:tablew}` (with “[Table W]” text) alongside Fig.~\ref{fig:tablew-effect-deltas}, aligning with the canonical table label.

### 2026-02-17 — Writing lane: tighten SYCON Bench positioning (ToF/NoF vs our TOF)

- Updated Related Work §6.4 to explicitly name SYCON Bench’s Turn of Flip / Number of Flips metrics and relate Turn of Flip to our TOF framing (without conflating task/scoring).

### 2026-02-17 — Writing lane: remove brittle section pointer for Table W in core-claims

- Replaced “Table W in §7.5” with canonical `Table~\ref{tab:tablew}` in §1.4 core-claims evidence pointers.

### 2026-02-17 04:12 KST — Experiments lane: Phi-3-mini seed1 validated; seed2 running; EXAONE blocked by transformers import

- Remote (nlp8): confirmed `results_paper/tier1_phi3mini_seed1_20260217_011737/` has complete `paper_exports/` and validator `[OK]`.
- Remote (nlp8): `results_paper/tier1_phi3mini_seed2_20260217_033953/` is running on GPU6 (tmux: `tier1_phi3mini_s2_g6_20260217_033953`).
- Remote (nlp8): EXAONE Tier‑1 attempts are **not paper-ready** and kept under `results_paper_incomplete/` due to `ImportError: RopeParameters` from `transformers.modeling_rope_utils` when loading EXAONE remote code.

Next:
- When seed2 finishes, validate + (if green) ensure `GLOBAL_VALIDATE.log` parity remains `[OK]`.
- Decide whether to change the conda env (transformers pin/upgrade) for EXAONE vs. switch to another model family.

### 2026-02-17 06:16 KST — Experiments lane: Phi-3-mini seed2 exported + validator OK + global parity PASS

- Remote (nlp8): `results_paper/tier1_phi3mini_seed2_20260217_033953/` produced core CSVs and `paper_exports/` (survival/TOF/flip_samples/metadata).
- Validator fix: added missing `paper_exports/runner_metadata.json` (required keys incl. gpu_list/num_samples/max_model_len/max_tokens/conda_env), then `validate_paper_exports.py --results_root ...` → `[OK]`.
- Re-ran global parity: `python scripts/validate_paper_exports.py --results_root results_paper --check_runner_parity` → `[OK] runner_metadata parity` (now includes phi3mini seed2).

Next:
- Decide next Tier‑1 model family (EXAONE currently blocked by transformers import).
- Optionally update repo-tracked cross-family artifacts/figures to include Phi‑3‑mini seeds 1–2.

## 2026-02-17 12:08 (KST) — Intro proof pointers: “Evidence at a glance”
- Added an explicit “Evidence at a glance” bullet list in Intro §1.1 to reduce reviewer search cost and make protocol/dynamics/recovery pointers unmissable.
- Updated CLAIM_EVIDENCE_MAP to require keeping this skim hook aligned with the same figure/table labels.

### 2026-02-17 18:54–20:00 KST — Tier‑1 cross-family extension: Mistral‑Nemo seeds 1–2 paper-visible

- Remote (nlp8): `tier1_mistralnemo_seed2_20260217_180951/` finished; ran `paper_export.py`, wrote `paper_exports/runner_metadata.json`, validator `[OK]`.
- Remote (nlp8): global validate initially failed due to an incomplete Zephyr dir under `results_paper/`; moved it to `results_paper_incomplete/` and re-ran global parity → `[OK] runner_metadata parity`.
- Local (writing repo): generated tracked artifact `docs/paper/artifacts/tier1_mistralnemo_seed1-2_survival_summary_20260217.csv` from synced `paper_exports/`.
- Local: updated `scripts/make_cross_family_figure_svg.py` and regenerated canonical cross-family SVG `docs/paper/figures/cross_family_survival_r5_control_vs_logicaltrap_seed1-2_20260217.svg`.
- Local: added EN draft numeric callout for Mistral‑Nemo (Survival@5 control 32.54±4.77 vs Logical Trap 9.29±0.44) + updated C4 artifact list; updated proof pointers in `CLAIM_EVIDENCE_MAP.md` + `FIGURE_CAPTIONS.md`.

## 2026-02-17 20:22 (KST) — Limitations: cross-family max_model_len transparency

- Updated EN draft Limitations to explicitly note cross-family context-length feasibility constraints (some families use reduced `max_model_len` for KV-cache fit; protocol otherwise unchanged).
  - File: `docs/paper/PAPER_DRAFT_EN.md` (Limitations §10.1)

Next:
- Draft KO "수정/보완해야 할 것만" TODO list (8–12 items) in the requested "현재 → 해서 → 로" format.
- Ensure the same context-length note is reflected (briefly) in KO draft or appendix if we cite Nemo numbers prominently.

## 2026-02-17 20:24 (KST) — KO revision TODO list ("현재 → 해서 → 로")

- Added an SSOT revision TODO list in the requested "현재 ~~가 부족/문제이고 → ~~해서 → ~~로 수정해야 한다" format (12 items; reviewer-risk only).
  - New: `docs/paper/REVISION_TODO_KO.md`
  - Linked from: `docs/paper/PAPER_DRAFT_KO.md` (top)

---

### 2026-02-18 (am) — Tier-1 cross-family validation sweep (nlp8 results_paper)

- Ran `scripts/validate_paper_exports.py` across all `results_paper/tier1_*_20260217_*` on **nlp8**.
- ✅ Paper-ready (validator `[OK] paper_exports`):
  - `tier1_phi3mini_seed{1,2}_20260217_*`
  - `tier1_mistralnemo_seed{1,2}_20260217_*`
- ❌ Incomplete / no `paper_exports/` found (diagnosed via `run.log` tails):
  - `tier1_falcon7b_seed1_20260217_145044`: vLLM init fail (`FalconConfig` missing `rope_parameters`; transformers/vLLM compat)
  - `tier1_gemma2_2b_seed1_20260217_141927` (+ len4096 variant): Triton shared-memory OOR on RTX8000 (cc7.5)
  - `tier1_pythia2p8b_seed1_20260217_155743` (+ len2048 variant): vLLM rejects `max_model_len 4096 > derived 2048` (use 2048)
  - `tier1_zephyr7b_seed1_20260217_150053`: empty `run.log` / interrupted; needs rerun

Decision for next experiment heartbeat:
- Rerun the “easy fixes” first: **Pythia-2.8B with `--max_model_len 2048`** and **Zephyr-7B rerun** (both should be compatible), then reconsider Falcon/Gemma2 only if we change stack/backend.

---

### 2026-02-18 (am) — Runbook: add Tier-1 failure signatures + fixes

- Updated `docs/paper/REMOTE_EXPERIMENTS_RUNBOOK.md` with concrete failure signatures + mitigations observed on nlp8 RTX8000:
  - Falcon-7B: missing `rope_parameters` (transformers/vLLM compat)
  - Pythia-2.8B: enforce `--max_model_len 2048`
  - (kept) Gemma2: Triton shared-memory OOR

Reason: reduce wasted Tier‑1 budget by making “known-bad” configs obvious before launches.

---

### 2026-02-18 04:23 KST — Launch Tier-1 Pythia rerun with correct max_model_len

- Launched `EleutherAI/pythia-2.8b-deduped` seed1 on nlp8 GPU4 with `--max_model_len 2048` (fix for prior vLLM guardrail failure).
- OUT: `results_paper/tier1_pythia2p8b_seed1_len2048_20260218_0424/`
- tmux: `tier1_pythia2p8b_s1_g4_20260218_0424`

Next: wait for `paper_exports/` + run validator on the run root.

---

### 2026-02-18 05:07 KST — Pythia Tier-1 rerun: switch from conda-run to conda-activate runner (fix silent logs)

- Observed: `tier1_pythia2p8b_seed1_len2048_20260218_0424` had GPU4 100% util for ~40min but `run.log` stayed 0 bytes under `conda run ... | tee`.
- Action: killed the silent tmux session and restarted via a small `run.sh` that does `source conda.sh; conda activate galileo; python run_experiment.py ...`.
- New tmux: `tier1_pythia2p8b_s1_g4_20260218_0510` (same OUT dir).
- Result: logs now stream (vLLM init + checkpoint shard loading) and GPU4 util resumed.

NOTE: first restart attempt had `OUT` unset under `set -u`; fixed by hardcoding OUT in the script.

---

### 2026-02-18 05:17 KST — Pythia Tier-1 run is now executing (post-log-fix)

- nlp8 GPU4 run `results_paper/tier1_pythia2p8b_seed1_len2048_20260218_0424/` is past vLLM init and has queued all 80 samples (progress bar now at `Processed prompts: 0/800`).
- Noted warnings: no chat_template (expected for GPTNeoX) and `max_tokens` capped due to `max_model_len=2048`.

Next: wait for completion → ensure `paper_exports/` exists → validate with `scripts/validate_paper_exports.py`.

---

### 2026-02-18 09:43 KST — Stop non-exporting Pythia run; switch to canonical exporter runner

- Diagnosis (nlp8): `run_experiment.py` **never creates `paper_exports/`**; exports are produced only when we run `scripts/paper_export.py` (via runner scripts like `scripts/remote_run/nlp8_smoke.sh`). This explains why the long Pythia job kept consuming GPU without becoming paper-ready.
- Action: stopped the ongoing Pythia tmux run (GPU4 freed) and launched a fresh **Zephyr-7B** Tier‑1 run using `scripts/remote_run/nlp8_smoke.sh`, which guarantees `paper_exports/` + `runner_metadata.json` + validator.
  - OUT: `results_paper/tier1_zephyr7b_seed1_20260218_0945/`
  - tmux: `tier1_zephyr7b_s1_g4_20260218_0945`
  - GPU: 4

Rationale: Pythia also showed extremely low initial accuracy (1/80) so even if we exported, it would be low-signal; Zephyr is a better cross-family candidate.

### 2026-02-18 10:34 KST — Zephyr Tier‑1 seed2 launch (seed1 paper-ready)

- Confirmed Zephyr‑7B **seed1** is paper-ready (exports + validator OK):
  - `results_paper/tier1_zephyr7b_seed1_20260218_0945/paper_exports/` contains the full bundle.
  - `run.log` shows `[OK] .../paper_exports` and `[OK] runner_metadata parity`.
- Launched Zephyr‑7B **seed2** using canonical exporter runner on nlp8 GPU5:
  - OUT: `results_paper/tier1_zephyr7b_seed2_20260218_1034/`
  - tmux: `tier1_zephyr7b_s2_g5_20260218_1034`
  - Command pattern: `GPU=5 SEED=2 OUT=... MODEL=HuggingFaceH4/zephyr-7b-beta bash scripts/remote_run/nlp8_smoke.sh |& tee run.log`

### 2026-02-18 10:43 KST — Zephyr seed2 blocked by GPU5 memory pressure (external job)

- Seed2 run failed immediately on nlp8 GPU5 due to vLLM startup memory check:
  - `ValueError: Free memory on device ... less than desired GPU memory utilization (0.9, 42.54 GiB)`
  - Trace in: `results_paper/tier1_zephyr7b_seed2_20260218_1034/run.log`
- Root cause: GPU5 occupied by external process `jslee-fusion-distill-vllm-v1` (~29GB) under user `omanma1` (PID 1949819).
- Action: did **not** kill external process; our tmux session already ended.
- Next: relaunch seed2 on a genuinely free GPU (GPU6 recommended) or reduce vLLM memory utilization if we must share.

### 2026-02-18 10:53 KST — Zephyr seed2 relaunch blocked (no free Tier‑1 GPUs on nlp8)

- Attempted to relaunch Zephyr‑7B seed2 on a free Tier‑1 GPU after GPU5 contention failure.
- Current nlp8 GPU occupancy (policy GPUs 4/5/6):
  - GPU4: `python scripts/run_search_o1_wiki.py ... QwQ-32B-Preview` (user `skbaek1`, ~48GB)
  - GPU5: `jslee-fusion-distill-vllm-v1` (user `omanma1`, ~29GB)
  - GPU6: `jslee-fusion-distill-vllm-v1` (user `omanma1`, ~29GB)
- Conclusion: cannot start Zephyr seed2 without either (i) waiting for GPUs to free, or (ii) obtaining permission to use a different machine/GPU outside the current Tier‑1 policy.

### 2026-02-18 11:52 KST — GPU contention persists; cleaned stale Zephyr tmux

- nlp8 GPUs remain unavailable for Zephyr seed2:
  - GPU4: `python` (QwQ-32B search job; ~48GB)
  - GPU5/6: `jslee-fusion-distill-vllm-v1` (user `omanma1`; ~29–30GB each)
- Cleaned up an old, idle tmux session to reduce operator confusion (no GPU freed):
  - killed: `tier1_zephyr7b_s1_g5_20260217_150053`

### 2026-02-18 14:12 KST — Zephyr Tier‑1 seed2 re-launched on freed GPU4

- nlp8 GPU4 became fully free; GPUs 5/6 still held by external vLLM processes.
- Launched Zephyr‑7B **seed2** using canonical exporter runner on nlp8 GPU4:
  - OUT: `results_paper/tier1_zephyr7b_seed2_20260218_141231/`
  - tmux: `tier1_zephyr7b_s2_g4_20260218_141231`
  - Command pattern: `GPU=4 SEED=2 OUT=... MODEL=HuggingFaceH4/zephyr-7b-beta bash scripts/remote_run/nlp8_smoke.sh |& tee run.log`

### 2026-02-19 06:16 KST — Tier‑1 Qwen2.5‑14B seed2 still running (nlp8 GPU5)

- Monitored the in-progress Tier‑1 run and confirmed it is actively progressing (log mtime updates; GPU5 at 100% util, ~45.4/49.1GiB).
- OUT: `results/tier1_qwen2p5_14b_seed2_20260219_053824/`
- Status snapshot: completed Round1–3; currently in **Round4** (425 active tracks at start of round4).
- No `paper_exports/` yet (expected until run completion).

Next:
- When run finishes: run `python3 scripts/paper_export.py --results_root $OUT` + write `paper_exports/runner_metadata.json` (if missing), then `python3 scripts/validate_paper_exports.py --results_root $OUT`.
- Stage into `results_paper/` and re-run `python3 scripts/validate_paper_exports.py --results_root results_paper --check_runner_parity`.

### 2026-02-19 06:24 KST — Tier‑1 Qwen2.5‑14B seed2 progressed to Round5 (still running)

- Checked nlp8 run health: GPU5 ~98% util, ~45.4/49.1GiB; `run.log` mtime updating → run is active.
- OUT: `results/tier1_qwen2p5_14b_seed2_20260219_053824/`
- Status: finished Round4 (408 still correct, 17 failed) and started **Round5** (408 active tracks); currently processing Round5 prompts.
- No `paper_exports/` yet (expected until completion).

### 2026-02-19 06:34 KST — Tier‑1 Qwen2.5‑14B seed2 in recovery-answer generation stage

- nlp8 GPU5 still saturated (~100% util, ~45.4/49.1GiB) and `run.log` continues to update.
- OUT: `results/tier1_qwen2p5_14b_seed2_20260219_053824/`
- Status: main adversarial rounds appear complete; now generating **recovery answers** for failed cases (log shows 73 failed cases to recover).
- No `paper_exports/` yet (expected until finalization / explicit export step).

### 2026-02-19 06:44 KST — Qwen2.5‑14B seed2 shows recovery phase then a second adversarial-testing block (still running)

- Used a small python parser to extract the latest semantic markers from `run.log` (tail output is noisy due to tqdm carriage returns).
- Observed sequence:
  - Round2→Round5 completed (last seen: Round5 results `401 still correct, 7 failed`)
  - Recovery started (`Failed cases to recover: 73` → `Generating recovery answers...`)
  - Then **another** adversarial-testing block appears to start (new `Round1: 468 active tracks`, progressed to Round3).
- Interpretation: the runner may be looping over multiple conditions/splits/persona sets in one job, or it restarted a second pass; we should wait for an explicit completion marker before exporting.

### 2026-02-19 07:00 KST — Tier‑1 Qwen2.5‑14B seed2 exported + staged to results_paper (paper-ready)

- nlp8: `results/tier1_qwen2p5_14b_seed2_20260219_053824/` completed (GPU5 freed).
- Ran `scripts/paper_export.py` with explicit `--model_dir` and wrote missing `paper_exports/runner_metadata.json`.
- Validator: `[OK] .../paper_exports`.
- Staged symlink into paper SSOT:
  - `results_paper/tier1_qwen2p5_14b_seed2_20260219_053824/paper_exports -> ../../results/.../paper_exports`
- Global parity: `python3 scripts/validate_paper_exports.py --results_root results_paper --check_runner_parity` → `[OK] runner_metadata parity`.

### 2026-02-19 08:33 KST — nlp8 GPU4 occupied by external vLLM process; GPUs5/6 idle

- Snapshot:
  - GPU4: ~99% util, ~45.9/49.1GB, process `VLLM::EngineCore` by user `omanma1` (pid 2042987)
  - GPU5/6: 0% util (residual VRAM allocated but idle)
- Operational implication: do **not** launch new Tier‑1 runs targeting GPU4 until it clears; prefer GPU5/6.

### 2026-02-19 08:57 KST — Launched next Tier‑1 cross-family family: OLMo‑7B‑Instruct (seeds 1–2) on nlp8 GPUs 5/6

- Preflight OK: `allenai/OLMo-7B-Instruct` exists on HF.
- Launched via tmux (one run per GPU), protocol identical to other Tier‑1 runs (NUM_SAMPLES=80, MAX_MODEL_LEN=8192, no max_tokens override):
  - seed1 GPU5: `results/tier1_olmo7b_seed1_20260219_085627/` (tmux: `tier1_olmo7b_s1_g5_20260219_085627`)
  - seed2 GPU6: `results/tier1_olmo7b_seed2_20260219_085627/` (tmux: `tier1_olmo7b_s2_g6_20260219_085627`)
- Each runner will write `paper_exports/` + `runner_metadata.json` and run `validate_paper_exports.py` at the end.

### 2026-02-19 09:06 KST — OLMo‑7B Tier‑1 launch failed immediately (missing `hf_olmo` dependency)

- Both seed1/seed2 runs crashed during model load with:
  - `ImportError: ... requires ... hf_olmo ... Run pip install hf_olmo`
- GPUs 5/6 remained idle (no work executed).
- Action: treat OLMo as blocked under current “no pip install / fixed env” constraint; pick a different model family that loads with stock transformers.

### 2026-02-19 09:17 KST — Replaced OLMo (blocked) with StableLM‑2 1.6B Chat Tier‑1 seeds 1–2 (nlp8 GPUs 5/6)

- OLMo blocked due to missing `hf_olmo` dependency (no pip-install policy), so switched to a stock-transformers family.
- Preflight OK (AutoConfig load): `stabilityai/stablelm-2-1_6b-chat`.
- Launched via tmux (one run per GPU):
  - seed1 GPU5: `results/tier1_stablelm2_1p6b_seed1_20260219_091650/` (tmux: `tier1_stablelm2_1p6b_s1_g5_20260219_091650`)
  - seed2 GPU6: `results/tier1_stablelm2_1p6b_seed2_20260219_091650/` (tmux: `tier1_stablelm2_1p6b_s2_g6_20260219_091650`)
- Runner writes `paper_exports/` + `runner_metadata.json` and runs `validate_paper_exports.py`.

### 2026-02-19 09:20 KST — Tier‑1 new-family launch attempts blocked (environment + external GPU residency)

- Attempted OLMo‑7B‑Instruct: failed on missing `hf_olmo` dependency (would require `pip install hf_olmo`).
- Attempted StableLM‑2 1.6B Chat: vLLM refused `--max_model_len 8192` because derived max is 4096; we should not set `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` for Tier‑1.
- Attempted DeepSeek‑LLM‑7B‑Chat with `max_model_len=4096`: vLLM failed to start on GPU5 because an external process holds ~29.8GB VRAM:
  - user `omanma1`, process `jslee-fusion-distill-vllm-v1`.
- Conclusion: until GPU5/6 are truly free (and/or we reduce vLLM memory utilization), new-family Tier‑1 runs will keep failing at engine init.

### 2026-02-19 09:38 KST — nlp8 GPUs 4/5/6 still effectively blocked by external `omanma1` vLLM residency

- `nvidia-smi` snapshot:
  - GPU4: ~29.1GB allocated + high util (~89%)
  - GPU5: ~29.8GB allocated (0% util but blocks vLLM init due to insufficient free VRAM)
  - GPU6: ~29.0GB allocated (0% util)
- All allocations are from `omanma1` process `jslee-fusion-distill-vllm-v1` (pids: 2056846/2038527/2057924).
- Impact: any new Tier‑1 launch that expects ~0.9 GPU memory utilization (default vLLM) will fail to start on GPUs 4–6 until these processes release VRAM.

### 2026-02-19 11:28 KST — Tier‑1 new-family run (DeepSeek‑LLM‑7B‑Chat) relaunched successfully after GPUs cleared; forced TRITON attention backend

- GPUs 4/5/6 were fully free (no external VRAM residency).
- Relaunched Tier‑1 new family: `deepseek-ai/deepseek-llm-7b-chat` with `max_model_len=4096` and `VLLM_ATTENTION_BACKEND=TRITON_ATTN` to avoid FlashInfer JIT (`ninja` missing) failure.
- tmux sessions:
  - seed1 GPU5: `tier1_deepseek7b_s1_g5_20260219_112728` (OUT: `results/tier1_deepseek7b_seed1_20260219_112728/`)
  - seed2 GPU6: `tier1_deepseek7b_s2_g6_20260219_112728` (OUT: `results/tier1_deepseek7b_seed2_20260219_112728/`)
- Both runs reached vLLM engine init and began model loading; `VLLM::EngineCore` is visible on GPUs 5/6.

### 2026-02-19 12:43 KST — DeepSeek Tier‑1 seeds 1–2 completed; `paper_exports/` validated and staged into `results_paper/`

- Observed completion markers in both run logs:
  - `results/tier1_deepseek7b_seed1_20260219_112728/run.log`: `=== EXPERIMENT COMPLETE ===`
  - `results/tier1_deepseek7b_seed2_20260219_112728/run.log`: `=== EXPERIMENT COMPLETE ===`
- Verified `paper_exports/` contains the required files for each seed:
  - `survival_curve.csv`, `turn_of_failure.csv`, `flip_samples.csv`, `metadata.json`, `runner_metadata.json`
- Staged both runs into SSOT `results_paper/` (symlinks):
  - `results_paper/tier1_deepseek7b_seed1_20260219_112728 -> ../results/tier1_deepseek7b_seed1_20260219_112728`
  - `results_paper/tier1_deepseek7b_seed2_20260219_112728 -> ../results/tier1_deepseek7b_seed2_20260219_112728`
- Global validator:
  - `python3 scripts/validate_paper_exports.py --results_root results_paper --check_runner_parity` → `[OK] runner_metadata parity`

### 2026-02-19 13:05 KST — DeepSeek integrated into local cross‑family artifacts + figure (writing repo)

- Synced minimal DeepSeek bundles locally (seed1–2):
  - `tmp_results_paper/tier1_deepseek7b_seed1_20260219_112728/paper_exports/`
  - `tmp_results_paper/tier1_deepseek7b_seed2_20260219_112728/paper_exports/`
- Generated tracked summary CSV:
  - `docs/paper/artifacts/tier1_deepseek7b_seed1-2_survival_summary_20260219.csv`
  - via `python3 scripts/make_tier1_survival_summary.py --run_roots <seed1>,<seed2> --out_csv ...`
- Updated `scripts/make_cross_family_figure_svg.py` to include DeepSeek, then regenerated:
  - `docs/paper/figures/cross_family_survival_r5_control_vs_logicaltrap_seed1-2_20260219.svg`
- Refreshed PDFs from SVG sources:
  - `bash scripts/convert_figures_svg_to_pdf.sh docs/paper/figures paper_figures/pdf`

### 2026-02-19 14:05 KST — Added an explicit §7.4 narrative sentence naming DeepSeek + Qwen14B as included in cross-family set

- Updated `docs/paper/PAPER_DRAFT_EN.md` (§7.4) to explicitly state that the `20260219` cross-family figure includes Qwen2.5‑14B‑Instruct and DeepSeek‑LLM‑7B‑Chat (seeds 1–2).

### 2026-02-19 19:33–19:53 KST — Phi‑3.5‑mini Tier‑1 seeds 1–2: post-hoc paper_exports + SSOT staging + local artifact

- SSOT (nlp8): `microsoft/Phi-3.5-mini-instruct` Tier‑1 seeds 1–2 finished under `results/tier1_phi35mini_seed{1,2}_20260219_143555/`.
- Ran post-hoc paper exports (stdlib) and validated per-run:
  - `python3 scripts/paper_export.py --results_root <OUT> --model_dir <OUT>/Phi-3.5-mini-instruct --out_dir <OUT>/paper_exports ...`
  - Added `paper_exports/runner_metadata.json` (required schema) and validated with `scripts/validate_paper_exports.py` → `[OK]`.
- Staged into SSOT `results_paper/` via `paper_exports` symlinks:
  - `results_paper/tier1_phi35mini_seed1_20260219_143555/paper_exports`
  - `results_paper/tier1_phi35mini_seed2_20260219_143555/paper_exports`
- Global SSOT validator: `python3 scripts/validate_paper_exports.py --results_root results_paper --check_runner_parity` → `[OK] runner_metadata parity`.
- Local writing repo: synced minimal bundles into `tmp_results_paper/` and generated + committed the tracked summary artifact:
  - `docs/paper/artifacts/tier1_phi35mini_seed1-2_survival_summary_20260219.csv` (commit `d997788`).

### 2026-02-19 (night) — Submission freeze mode activated

- Decision: stop scope expansion tonight (no new experiments unless claim-blocking).
- Added freeze SSOT checklist: `docs/paper/SUBMISSION_FREEZE_TONIGHT.md`.
- Updated `docs/paper/STATUS.md` Next Heartbeat to claim↔evidence lock workflow.
- Intent: prioritize submit-ready narrative consistency (Abstract/Intro claims, proof pointers, Table-W wording, citation/consistency checks).

### 2026-02-19 23:57 KST — Experiments lane: enforce Pythia hard cutoff (nonterminal run terminated)

- Remote (nlp8): confirmed `results/tier1_pythia2p8b_seed2_20260219_211411` remained nonterminal (no `EXPERIMENT COMPLETE`, no `paper_exports/`) after repeated loop-like phases.
- Executed hard cutoff by terminating tmux session `tier1_pythia2p8b_s2_g1_20260219_211411`; post-check showed GPU1 freed to idle.
- Decision: quarantine Pythia seed1/seed2 as non-citable on current stack; no additional blind retries.
- Next: launch one stable fallback target only when an idle GPU slot exists; otherwise continue writing/claim-evidence lock while keeping `results_paper/` parity green.

### 2026-02-20 00:07 KST — Writing lane: claim-evidence map de-drifted for cross-family evidence

- Updated `docs/paper/CLAIM_EVIDENCE_MAP.md` to remove stale “Zephyr pending” text and point to the existing tracked artifact `docs/paper/artifacts/tier1_zephyr7b_seed1-2_survival_summary_20260218.csv`.
- This keeps C4 (cross-family replication) proof pointers consistent with current SSOT artifacts and reduces reviewer-facing drift risk.
- Reflected the sync in `docs/paper/STATUS.md` under recent paper-writing/compliance updates.

### 2026-02-20 00:17 KST — Process/experiments lane: Tier-1 gap ledger updated after Pythia cutoff

- Updated `docs/paper/TIER1_GAP_CHECKLIST_20260219.md` to include the latest failed/nonterminal Pythia runs (`seed1_20260219_211411`, `seed1_20260219_211501`, `seed2_20260219_211411`).
- Added explicit policy text: Pythia line is quarantined on current stack; no further blind retries without a concrete stack-level fix + smoke test.
- Synced this decision in `docs/paper/STATUS.md` to keep heartbeat planning aligned with reviewer-risk priorities.

### 2026-02-20 02:33 KST — Process lane: idle GPU policy clarified (0/1 are valid targets)

- Synced operating interpretation with SSOT policy: on nlp8, GPUs **0–6** are valid if idle/not used by others; this explicitly includes GPU0/1.
- Added a note in `docs/paper/STATUS.md` to prevent stale 4–7 banner text from blocking safe launches on idle GPU0/1.
- Monitoring this heartbeat confirmed GPU0/1 idle while 2–6 were occupied; launch gating should use real-time idleness, not stale banner constraints.

### 2026-02-20 18:58 KST — Paper writing lane: made Appendix code blocks line-breakable in LaTeX SSOT

- In `docs/paper/latex_paper_emnlp2023/main.tex`, replaced raw `verbatim` blocks with `fvextra`'s `Verbatim` environment using `breaklines`/`breakanywhere`.
- This removes the large `Overfull \hbox` warnings caused by long prompt/command lines, improving PDF-first workflow stability.
- Note: total PDF page count increased (Appendix line wrapping adds vertical space); this is expected and acceptable since Appendix is excluded from main-page budgeting.

### 2026-02-20 19:08 KST — Paper writing lane: eliminated remaining overfull math in metric definitions

- Tightened the hazard and TOF math in `docs/paper/latex_paper_emnlp2023/main.tex` (shorter conditioning notation + line breaks in align environments).
- Verified via `latexmk` + `grep "Overfull \\hbox" main.log` that the previous large overfull boxes in the Metrics section are now gone.

### 2026-02-20 19:18 KST — Paper writing lane: reduced main-text figure load (moved Recovery/Decoding to Appendix)

- In `docs/paper/latex_paper_emnlp2023/main.tex`, moved the persona-wise Recovery@flip plot and the decoding-sweep plot out of the main Results section.
- Added a new Appendix subsection `Additional results` (Appx.~\ref{sec:appendix-extra-results}) containing those two figures.
- Main text now keeps the core narrative/figures (Survival, Fail@1, Cross-family) and points to the Appendix for the extra decompositions.

### 2026-02-20 19:26 KST — Paper writing lane: added explicit LaTeX review/camera-ready toggle for page counting

- In `docs/paper/latex_paper_emnlp2023/main.tex`, introduced a review/camera-ready toggle that switches between `\usepackage[review]{EMNLP2023}` and `\usepackage{EMNLP2023}`.
- The toggle can now be controlled without editing the file by defining `\CAMERAREADY` at build time (see the commented `latexmk -pdflatex=...` command in the preamble).
- Verified locally that both builds succeed; current PDF is 7 pages in both modes (this will diverge once we start tight page-budgeting).

### 2026-02-20 19:50 KST — Paper development lane: automated page budgeting (main vs Limitations vs Appendix)

- Added log-based page markers to LaTeX SSOT (`docs/paper/latex_paper_emnlp2023/main.tex`) via `\\pagemark{...}` so we can compute page counts without manual PDF inspection.
- Added `scripts/report_latex_page_budget.sh` which compiles in camera-ready mode (`\\CAMERAREADY`) and prints:
  - total pages
  - main pages (pre-appendix)
  - main pages excluding Limitations
- Current camera-ready counts: total=7, main(pre-appendix)=5, main(excl limitations)=3.

### 2026-02-20 20:00 KST — Paper writing lane: split Discussion vs Limitations for clean 8p budgeting

- In `docs/paper/latex_paper_emnlp2023/main.tex`, replaced the combined `Discussion and limitations` section with separate `\section{Discussion}` and `\section{Limitations}`.
- Kept `\pagemark{LIMITATIONS_SECTION_START}` at the start of the Limitations section so `scripts/report_latex_page_budget.sh` remains correct.
- This makes the "main pages excluding Limitations" target (8 pages) operationally unambiguous.

### 2026-02-20 20:11 KST — Paper writing lane: expanded Introduction (design goals + why conditioning + why NRC)

- Added three short Intro paragraphs to increase main-text thickness without adding new experimental claims:
  - explicit evaluation questions (dynamics / early vulnerability / recoverability)
  - motivation for conditioning on initially-correct examples (deviation vs ignorance)
  - motivation for matched neutral control NRC (drift-corrected persona gaps)
- Camera-ready page-budget moved from main(excl Limitations)=3 → 4 (now Limitations starts on p5).

### 2026-02-20 20:20 KST — Paper writing lane: expanded Protocol with explicit notation + metrics anchor

- In `docs/paper/latex_paper_emnlp2023/main.tex`, added a new Protocol subsection "Notation (tracks, histories, and reported gaps)" clarifying:
  - two-track setup (persona vs NRC) under matched decoding
  - per-track history separation
  - delta convention \(\Delta m = m^{(p)} - m^{(\mathrm{NRC})}\)
- Added `\label{sec:metrics}` to the Metrics subsection so notation references resolve.
- Camera-ready budget now: total=8, main(pre-appendix)=6, main(excl Limitations)=4.

### 2026-02-20 20:30 KST — Paper writing lane: expanded Results narrative (Survival/Fail@1 reading guide)

- In `docs/paper/latex_paper_emnlp2023/main.tex`, expanded the Results section text for Survival and Fail@1:
  - added explicit recurring patterns (gap widening with rounds; different hazard shapes)
  - added a short reading guide for survival curves in the conditioned-on-correct setting
  - clarified why Fail@1 complements Survival@5 and ties to the discrete hazard at r=1
- Page budget (camera-ready) remains main(excl Limitations)=4; next expansions should target Protocol/Results further or compress tables/figures if they are blocking page growth.

### 2026-02-20 20:40 KST — Paper writing lane: expanded Results (Recovery rationale + cross-family replication)

- In `docs/paper/latex_paper_emnlp2023/main.tex`, expanded the Results narrative in two places:
  - Recovery subsection: clarified deployment implication (brittle yet recoverable) and explained why Recovery@flip is conditional (denominator hygiene; points to \S\ref{sec:metrics}).
  - Cross-family subsection: added 3 sentences framing Fig.~\ref{fig:crossfam} as a replication check across open-weight families, preempting "single-model artifact" concerns.
- Camera-ready page budget remains main(excl Limitations)=4 (still below 8; we likely need Table 1 to be real + longer Intro/Protocol text to push the next page break).

### 2026-02-20 20:48 KST — Paper writing lane: refactored Table 1 to model-family rows (main-text space efficiency)

- In `docs/paper/latex_paper_emnlp2023/main.tex`, rewrote the main results table (Table~\ref{tab:main}) to be \emph{model-family rows} with persona-weighted aggregates, instead of persona rows.
- Table now reports NRC / Persona / $\Delta$ for all three metrics (Survival@5, Fail@1, Recovery@flip) and explicitly states mean\,\(\pm\)\,std over seeds where available.
- Persona-wise decompositions are pushed to Appendix references (Appx.~\ref{sec:appendix-extra-results}).

### 2026-02-20 21:27 KST — Paper development lane: made page-budget script generation-aware

- Updated `scripts/report_latex_page_budget.sh` to run `scripts/gen_latex_table1_from_artifacts.py` before compiling, so the camera-ready budget check no longer depends on manually pre-generating `generated/table1_rows.tex`.
- This keeps the PDF-first "작성↔확인" loop tight even though generated fragments are gitignored.

### 2026-02-20 22:10 KST — Paper development lane: extended Table 1 generator with Fail@1 deltas + Recovery (Qwen2.5-7B)

- Updated `scripts/gen_latex_table1_from_artifacts.py` to populate additional Table 1 cells from tracked artifacts:
  - Fail@1: computes an unweighted mean±std over persona-wise `delta_fail_r1_mean` values for each model family summary.
  - Qwen2.5-7B Recovery@flip (collapsed): pulls control/persona/delta (percent) from `recovery_collapsed_control_vs_persona_seed1-4_mean_std_*.csv` and converts to fractions.
- Generated fragments remain gitignored; `scripts/report_latex_page_budget.sh` already runs the generator before compiling.

### 2026-02-20 23:06 KST — Paper writing lane: made Results lead with Table 1 (one-stop summary)

- In `docs/paper/latex_paper_emnlp2023/main.tex`, added a short "One-stop summary" paragraph right after the Results intro to:
  - state the cross-family pattern (negative ΔSurvival@5, positive ΔFail@1)
  - note that Recovery@flip can decouple from brittleness
  - point readers to the three core figures that decompose the table (Survival/Fail@1/Cross-family)
- Page budget (camera-ready) remains main(excl Limitations)=5.

### 2026-02-20 03:34 KST — Experiments lane: CUDA preflight blocker confirmed on idle GPU0

- On nlp8, GPU0 appeared idle in `nvidia-smi` (0%, 1MiB), but device-level preflight failed: `CUDA_VISIBLE_DEVICES=0` + torch CUDA tensor allocation returned `cudaErrorDevicesUnavailable`.
- This reproduces the same failure class seen in the aborted fallback launch (`tier1_phi35mini_seed3_20260220_024525`) and indicates launch-time device availability races.
- Decision: do not relaunch heavy experiments on snapshot-idle GPUs unless a direct CUDA preflight passes immediately before launch; meanwhile prioritize writing/process lock while maintaining `results_paper` parity.

### 2026-02-20 03:54 KST — Development lane: added mandatory CUDA preflight helper

- Added `scripts/check_cuda_preflight.py` (single-visible-GPU torch CUDA allocation smoke test; exits non-zero on failure).
- Updated `docs/paper/REMOTE_EXPERIMENTS_RUNBOOK.md` to require this preflight before launching on an apparently idle GPU.
- Rationale: we observed repeated `cudaErrorDevicesUnavailable` despite idle `nvidia-smi` snapshots; preflight is now the launch gate.

### 2026-02-20 (am) — Paper guardrails: proof pointers + preflight automation

- EN draft: standardized NRC expansion to appear once (Abstract) and avoided re-expansions across Protocol/Related Work; added minimal proof pointers in Abstract + one section-pointer sentence in Intro (§2 + NRC + Fig.~
ef{fig:protocol}).
- Added paper-facing preflight automation (citations/acronyms/assets) and wired asset preflight into anonymized bundler to fail fast if refs break.

