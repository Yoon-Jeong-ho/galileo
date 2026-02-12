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
  - File: docs/paper/PAPER_DRAFT_EN.md

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
