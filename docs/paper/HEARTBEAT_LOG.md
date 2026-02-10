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
