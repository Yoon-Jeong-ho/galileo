# HEARTBEAT_LOG (rolling, truncated)

This file is intentionally kept short to reduce token burn.

- Full archive: `docs/paper/archive/HEARTBEAT_LOG_full_2026-02-24.md`
- Current SSOT plan/checklist: `docs/paper/EMNLP_2026_FEEDBACK_ACTION_PLAN.md`

---

- Checked `tmux ls` + `nvidia-smi` on nlp8.
- Found GPU3 is in use by our own unrelated job (aa007878 `Training/train_llama.py`); GPUs 0/1/2/4/5/6 idle.
- Decision: do **not** launch a new GALILEO run in this heartbeat to avoid CPU/IO contention while the training job is active; instead updated SSOT docs to reflect the 0–6 idle-only policy consistently.
### 2026-02-22 04:39 KST — Writing lane: fill Table 1 (Survival@5 + Fail@1) in the EN draft

- Used the auto-generated row values (from `scripts/make_table1_partial_from_results_paper_exports.py`) to replace the `--` placeholders for **Survival@5** and **Fail@1** in Table~\ref{tab:main}.
- Updated: `docs/paper/PAPER_DRAFT_EN.md` now has mean±std numbers for all 7 Table-1 rows (Survival@5 + Fail@1).
- Left **Recovery@flip** and **$n_0$/seed** as TODO placeholders until we add those fields to the table artifact.
### 2026-02-22 04:59 KST — Writing/dev lane: fill Table 1 n0/seed from results_paper exports

- Computed and tracked `n0` (avg initially-correct examples per seed) per run alias by reading `results_paper/*/paper_exports/survival_curve.csv` totals at round=1.
  - Added tracked artifact: `docs/paper/artifacts/table1_n0_from_results_paper_exports_20260222.csv`.
- Updated `scripts/make_table1_partial_from_results_paper_exports.py` to accept `--n0_csv` and emit the filled $n_0$/seed column.
- Updated `docs/paper/PAPER_DRAFT_EN.md` Table~\ref{tab:main} to fill the $n_0$/seed column (Recovery@flip remains TODO).
### 2026-02-22 05:29 KST — Writing/triage: explain Table 1 n0 variance (Mistral seed mismatch)

- Investigated why Table~\ref{tab:main} shows huge $n_0$/seed variance for Mistral-7B.
- Found `results_paper/mistral_seed1` has `survival_curve.csv` total=341 at round=1, while `mistral_seed2` total=875 (these totals are the initially-correct pool sizes).
- `metadata.json` indicates the two runs come from different result roots and even different `num_flip_samples` (50 vs 200), suggesting they were produced under mismatched configs / eval sets.
- Added a caption note clarifying that $n_0$/seed is reported to make such mismatches visible; recommended rerunning Mistral seeds 1–2 under the standardized Tier‑1 setting.
### 2026-02-22 05:39 KST — Writing lane: mark Mistral row as non-standard in Table 1

- Added a dagger marker to the Mistral-7B row in Table~\\ref{tab:main} and a corresponding caption note.
- Goal: prevent over-trusting cross-seed aggregates when $n_0$/seed indicates a strong mismatch (likely config divergence).
- Temporary guardrail until we rerun Mistral seeds 1–2 under the standardized Tier‑1 setting.
### 2026-02-22 05:59 KST — Dev lane: add recovery@flip extractor for Table 1 (results_paper -> artifact)

- Added `scripts/make_table1_recovery_from_results_paper.py` to compute Recovery@flip (NRC vs pooled persona pressure) for each `results_paper/<alias>`.
- It uses `paper_exports/metadata.json` to locate the original `results_root` and reads `recovery_accuracy.csv` (if present).
- Output is a tracked CSV artifact suitable for filling the remaining Recovery@flip block of Table~\ref{tab:main}.
### 2026-02-22 06:29 KST — Writing/dev lane: document Recovery@flip NRC-missing issue blocking Table 1

- Confirmed that our current recovery logs for paper-ready runs appear to omit `neutral_reask_control` rows in `recovery_accuracy.csv` (control_total=0 → NRC Recovery@flip NaN).
- Updated `docs/paper/PAPER_DRAFT_EN.md` to explicitly state this as the reason Recovery@flip cells are still TODO in Table~\ref{tab:main}.
- Updated `docs/paper/STATUS.md` with the two concrete resolution paths: (i) log recovery for NRC, or (ii) report recovery as persona-only with NRC marked N/A.
### 2026-02-22 06:39 KST — Dev lane: identify NRC label mismatch in recovery CSV (fix extractor)

- Investigated why NRC Recovery@flip was NaN when reading `recovery_accuracy.csv`.
- Found `run_experiment.py` writes persona names via `get_persona_name(...)`, so NRC appears as the **display label** `Control Re-asking` (not `neutral_reask_control`).
- Updated `scripts/make_table1_recovery_from_results_paper.py` to treat `Control Re-asking` as the NRC control persona, unblocking Table-1 recovery extraction.
### 2026-02-22 06:59 KST — Dev lane: add one-command restage for results_paper/

- Added `scripts/restage_results_paper_from_manifest.py` to rebuild the paper-only symlink root `results_paper/` from a simple CSV manifest (alias -> run_dir).
- Added template manifest: `docs/paper/results_paper_manifest.TEMPLATE.csv`.
- Updated `docs/paper/REMOTE_EXPERIMENTS_RUNBOOK.md` with the restage command (useful after `git clean -fd` on nlp8).
### 2026-02-22 07:09 KST — Process/dev lane: make results_paper manifest explicitly local-only

- Updated `docs/paper/REMOTE_EXPERIMENTS_RUNBOOK.md` to clarify that `docs/paper/results_paper_manifest.tsv` will contain absolute internal paths and should remain **local/untracked** (not for anonymized bundles).
- Created a local placeholder file `docs/paper/results_paper_manifest.tsv` and excluded it via `.git/info/exclude` so it won’t accidentally get committed.
### 2026-02-22 07:19 KST — Experiments/dev lane: restage results_paper (minimal) + extract Recovery@flip (NRC vs persona)

- On nlp8, rebuilt `results_paper/` (16 aliases; minimal set needed for Table 1 rows that still exist under `results/`) using `scripts/restage_results_paper_from_manifest.py` and validated it (`results_paper/GLOBAL_VALIDATE.log`).
- Ran `scripts/make_table1_recovery_from_results_paper.py` (with NRC label = `Control Re-asking`) to generate a per-alias Recovery@flip CSV.
- Synced the artifact into the local repo as: `docs/paper/artifacts/table1_recovery_from_results_paper_20260222.csv`.
- Note: some Tier‑1 families previously staged only under `results_paper/` (e.g., Phi‑3‑mini, Mistral‑Nemo, DeepSeek/Yi) were not in the current minimal restage set and will require adding their run roots to the manifest (or regenerating them) to fill their Recovery cells.
### 2026-02-22 07:29 KST — Writing/dev lane: fill Table 1 Recovery@flip (partial coverage)

- Filled Table~\ref{tab:main} Recovery@flip columns for the families covered by `docs/paper/artifacts/table1_recovery_from_results_paper_20260222.csv` (computed from nlp8 recovery logs).
- Updated `scripts/make_table1_partial_from_results_paper_exports.py` to accept `--recovery_csv` and auto-generate LaTeX rows including Recovery@flip (reported in %).
- Remaining TODO: add Phi‑3‑mini and Mistral‑Nemo run roots back into the nlp8 `results_paper` manifest so their Recovery rows can be extracted and Table~\ref{tab:main} can be fully filled.
### 2026-02-22 07:59 KST — Writing lane: clarify missing Recovery@flip entries in Table 1

- Added an explicit note in `docs/paper/PAPER_DRAFT_EN.md` Table~\\ref{tab:main} caption and the post-table provenance paragraph explaining that Recovery@flip cells can be `--` when the underlying `recovery_accuracy.csv` is not available under the currently staged paper SSOT.
- This avoids reviewer confusion about whether `--` means “not applicable” vs “not yet computed.”
### 2026-02-22 08:39 KST — Writing lane: mark cross-family Recovery@flip gaps explicitly in Table 1

- Added a $^{\ddagger}$ marker to Phi-3-mini and Mistral-Nemo rows in Table~\\ref{tab:main} and a matching caption note.
- Goal: make it unambiguous that missing Recovery@flip entries are a **known TODO** (missing recovery logs under the currently staged paper SSOT), not an omitted metric or a zero.
### 2026-02-22 09:39 KST — Writing lane: explicitly scope Table 1 Recovery@flip coverage

- Added a one-sentence clarification in `docs/paper/PAPER_DRAFT_EN.md` §7.0 that Recovery@flip is reported only where recovery logs are available; missing entries remain explicit TODOs.
- Goal: prevent reviewers from assuming missing recovery entries are zeros or from over-weighting incomplete cross-family recovery coverage.
### 2026-02-22 09:59 KST — Writing lane: explicitly list which Table-1 Recovery rows are TODO

- Updated `docs/paper/PAPER_DRAFT_EN.md` right below Table~\ref{tab:main} provenance note to explicitly list the current Recovery@flip TODO rows (Phi-3-mini, Mistral-Nemo).
- Goal: keep Option-B (partial recovery coverage) reviewer-proof by making the gap explicit and scoped.
### 2026-02-22 10:19 KST — Writing lane: add denominator caveat for Table-1 Recovery@flip

- Added an explicit interpretation note right below Table~\ref{tab:main} explaining that Recovery@flip is conditional on flipping and its effective sample size varies.
- Pointed to the tracked artifact columns (`control_total`, `persona_total`) in `docs/paper/artifacts/table1_recovery_from_results_paper_20260222.csv` to make this auditable.
### 2026-02-22 10:39 KST — Writing lane: add appendix note on how to complete missing Recovery@flip rows

- Added a short Appendix A note explaining why cross-family Recovery@flip coverage is partial and how to complete the TODO rows (rerun missing families with recovery logging, then regenerate the recovery artifact).
### 2026-02-22 10:49 KST — Writing lane: add appendix denominators note for Recovery@flip

- Added tracked artifact: `docs/paper/artifacts/table1_recovery_denominators_20260222.csv` summarizing recovery denominators (control_total/persona_total) per Table-1 row.
- Added Appendix A.1.1 note in `docs/paper/PAPER_DRAFT_EN.md` pointing readers to the denominator artifact to calibrate Recovery@flip variance.
### 2026-02-22 10:59 KST — Writing lane: add LaTeX appendix table for Recovery@flip denominators

- Added `scripts/make_table1_recovery_denominators_latex.py` to print LaTeX-ready rows from `docs/paper/artifacts/table1_recovery_denominators_20260222.csv`.
- Inserted a small LaTeX table snippet in Appendix A.1.1 of `docs/paper/PAPER_DRAFT_EN.md` that reports $|F|_{NRC}$ and $|F|_{persona}$ (flip-case denominators) so recovery variance is properly contextualized.
### 2026-02-22 11:09 KST — Writing/dev lane: make Recovery denominator appendix table auto-generated

- Added `scripts/gen_recovery_denominators_rows_tex.py` to generate `docs/paper/latex_paper_emnlp2023/generated/recovery_denominators_rows.tex` from the tracked artifact `docs/paper/artifacts/table1_recovery_denominators_20260222.csv`.
- Updated the Appendix LaTeX snippet in `docs/paper/PAPER_DRAFT_EN.md` to `\\input{generated/recovery_denominators_rows}` to eliminate drift between the CSV artifact and the paper table.
### 2026-02-22 11:29 KST — Writing/dev lane: best-effort LaTeX input for generated denominator rows

- Wrapped the appendix denominator table rows include with `\IfFileExists{...}{\input{...}}{...}` so local builds don’t hard-fail if the generated file is missing.
- CI/paper builds can still enforce fail-fast by ensuring the generator runs before LaTeX compile.
### 2026-02-22 12:09 KST — Experiments triage: GPU alloc preflight confirms GPU0 unusable

- On nlp8, ran `nvidia-smi` + `bash scripts/check_cuda_preflight_all.sh`.
- Result: GPU0 FAIL, GPUs1–6 OK (GPU3 still occupied by our PrivacyRestore training job).
- Operational decision: do not schedule GALILEO on GPU0 even if it appears idle; prefer GPUs1/2/4/5/6 (idle + preflight OK).
### 2026-02-22 12:50 KST — Dev lane: make Tier‑1 pilot command generation configurable

- Updated `scripts/print_crossfamily_run_commands.py` to accept overrides (`--num_samples`, `--max_model_len`, `--max_tokens`, `--greedy_temperature`) so we can generate a 50-sample seed1 pilot command without hand-editing.
- Added a short run note: `docs/paper/NEXT_TIER1_PILOT.md` (preflight + command generator + cap-check gate).

### 2026-02-24 14:37 KST — Dev lane: export normalized recovery_accuracy into paper_exports (Table‑1 unblock)

- Updated `scripts/paper_export.py` to optionally export `recovery_accuracy.csv` into `paper_exports/` when present, while normalizing persona ids so NRC is always `neutral_reask_control`.
- This unblocks Table~1 Recovery@flip auto-fill from the same SSOT `paper_exports/` bundle (no more per-run ad-hoc parsing) and removes the fragile 'Control Re-asking' display-name mismatch.

### 2026-02-24 14:47 KST — Dev lane: Table‑1 extractor now computes Recovery@flip when exported

- Extended `scripts/make_table1_from_results_paper_exports.py` to read `paper_exports/recovery_accuracy.csv` (when present) and emit `nrc_recovery`, `persona_recovery`, `delta_recovery` columns.
- Recovery columns are left blank for older bundles without recovery exports (safe partial progress).

### 2026-02-24 14:57 KST — Dev lane: LaTeX Table‑1 generator ingests Recovery from the same results_paper exports artifact

- Updated `scripts/gen_latex_table1_from_artifacts.py` to remove the hardcoded `model_to_aliases_for_recovery` mapping.
- Recovery cells are now filled (when available) by reading `docs/paper/artifacts/table1_from_results_paper_exports_*.csv` and taking mean±std over staged aliases/seeds, matching Fail@1 behavior.

### 2026-02-24 15:08 KST — Experiments/dev: backfill Mistral Recovery exports and regenerate Table‑1 artifact

- On nlp8 SSOT, re-ran `scripts/paper_export.py` for `results_paper/mistral_seed1` and `results_paper/mistral_seed2` so `paper_exports/recovery_accuracy.csv` is now present (NRC normalized).
- Regenerated tracked artifact `docs/paper/artifacts/table1_from_results_paper_exports_20260224.csv`, which now contains Recovery@flip columns for those Mistral seeds.
- Synced the artifact back to the local repo and committed it.

### 2026-02-24 15:18 KST — Planning lane: Recovery@flip pipeline now end-to-end; next is SSOT backfill for another family

- Confirmed the Recovery@flip pipeline is now end-to-end in code (export → Table1 artifact → LaTeX ingestion).
- Mistral seeds were backfilled first on nlp8; next step is to locate+backfill Recovery exports for one more Tier‑1 family (Phi‑3‑mini or Mistral‑Nemo) on nlp8, or explicitly mark as missing/needs rerun if run dirs no longer exist.

### 2026-02-24 15:29 KST — Experiments triage (nlp8): Phi‑3‑mini / Mistral‑Nemo run dirs not found

- Checked nlp8 `/data_x/aa007878/galileo/results` for `*phi3mini*` and `*mistralnemo*` at maxdepth=1: none found.
- Checked nlp8 `results_paper/` for aliases containing `phi|nemo`: none found.
- Only Phi‑3.5‑mini runs are currently present under `results/` (seed1–3). This means the “backfill recovery for Phi‑3‑mini/Mistral‑Nemo” step requires either recovering archived run dirs or rerunning those families.

### 2026-02-24 15:40 KST — Planning/dev: wrote Tier‑1 rerun runbook note (Phi‑3‑mini vs Mistral‑Nemo)

- Added `docs/paper/NEXT_TIER1_RERUN_PLAN.md` with a single-run tmux template + preflight/token-cap gates.
- Purpose: make the next Tier‑1 rerun decision executable immediately once the target family is chosen.

### 2026-02-24 16:00 KST — Planning: pick default next Tier‑1 rerun target (Phi‑3‑mini seed1)

- Updated `docs/paper/NEXT_TIER1_RERUN_PLAN.md` to recommend Phi‑3‑mini seed1 as the default next rerun target unless we explicitly want Nemo.
- Rationale: fastest path to restore missing-family Recovery@flip coverage in Table 1 with minimal GPU risk.

### 2026-02-24 16:15 KST — Docs: README updated (SSOT policy + Table‑1 regen)

- Updated repo root `README.md` to reflect the correct experiment SSOT (nlp8, GPUs 0–6 idle-only+preflight) and to add Table‑1 auto-regeneration commands from `results_paper/*/paper_exports`.

### 2026-02-24 16:10 KST — Writing/analysis: KO results analysis updated with SSOT + proof pointers

- Updated `docs/paper/PAPER_RESULTS_ANALYSIS_KO.md` to remove stale nlp16/raid6 references and to anchor analysis on nlp8 `results_paper/` + tracked artifacts.
- Added a “핵심 정량 결과” subsection that links Table W, cross-family figure, and the auto-extracted Table‑1 artifact CSV.

### 2026-02-24 16:20 KST — Docs: paper README now includes Table‑1 regen (reduces “table looks empty” risk)

- Updated `docs/paper/README.md` to include Table‑1 regeneration commands (`make_table1_from_results_paper_exports.py` + `gen_latex_table1_from_artifacts.py`).
- This makes the Table‑1 pipeline discoverable for collaborators and reduces the chance Table‑1 stays stale/partial.

### 2026-02-24 17:02 KST — Experiments: started Tier‑1 Phi‑3‑mini seed1 rerun (nlp8, GPU1)

- Launched `microsoft/Phi-3-mini-4k-instruct` seed1 standardized Tier‑1 rerun in tmux on nlp8.
- Session: `tmux attach -t tier1_phi3mini_seed1_rerun`
- Output dir: `results/tier1_phi3mini_seed1_rerun_20260224_170149/` (run.log inside)
- Gates in-script: CUDA alloc preflight → vLLM init preflight → run → paper_export (incl recovery) → runner_metadata → validator → token-cap check.

### 2026-02-24 18:12 KST — Experiments: restarted Phi‑3‑mini as a safer pilot (200 samples, max_tokens=512)

- Detected the previous Phi‑3‑mini seed1 rerun stalled (no new outputs after ~17:20 while GPU stayed busy). Sent SIGINT to stop cleanly and free GPU1.
- Restarted a safer pilot run in tmux with reduced load to avoid long-tail stalls:
  - Session: `tmux attach -t tier1_phi3mini_seed1_pilot`
  - Output: `results/tier1_phi3mini_seed1_pilot_200_512_20260224_181145/`
  - Params: `num_samples=200`, `max_tokens=512`, `max_model_len=4096`.
- Plan: if this pilot reaches paper_exports+validator OK, scale up to 1000 samples (seed1) and then seed2.

### 2026-02-24 19:01 KST — Experiments: fixed pilot benchmark coverage by forcing --data_dir

- Root cause: `config.DATA_DIR=/data_x/aa007878/galileo/data` currently contains only 4 datasets (`arc_easy_val_50.jsonl`, `squad_val_50.jsonl`, `gsm8k.jsonl`, `svamp.jsonl`). Without `--data_dir`, runs may implicitly use a partial default list, which makes pilot runs non-comparable to Tier‑1.
- Action: stopped the earlier pilot and restarted a new pilot that **forces dataset coverage** via `--data_dir /data_x/aa007878/galileo/data`.
  - Session: `tmux attach -t tier1_phi3mini_seed1_pilotfull`
  - Output: `results/tier1_phi3mini_seed1_pilotFULL_200_512_20260224_190053/`
  - Params: `num_samples=200`, `max_tokens=512`.

### 2026-02-24 19:50 KST — Planning: created SSOT feedback→action plan + heartbeat checklist

- Added `docs/paper/EMNLP_2026_FEEDBACK_ACTION_PLAN.md` capturing all received reviewer-style feedback as concrete counters + prioritized experiments.
- Includes a 10-min heartbeat checklist with tickable items so progress is cumulative and auditable.

### 2026-02-24 20:00 KST — Infra/data: created Tier‑1 6-benchmark data SSOT on nlp8

- User decision: **match Tier‑1 to the full 6-benchmark set** before proceeding.
- On nlp8, staged `data_tier1_6/` with symlinks to the 6 canonical JSONL files:
  - gsm8k, svamp, arc_easy_validation, squad11_validation, squad20_validation, triviaqa_rc_validation.
- Updated `docs/paper/EMNLP_2026_FEEDBACK_ACTION_PLAN.md` to require `--data_dir /data_x/aa007878/galileo/data_tier1_6` for all Tier‑1 runs.

### 2026-02-24 20:00 KST — Runbook: enforce Tier‑1 data_tier1_6 + conda gates on nlp8

- Updated `docs/paper/NEXT_TIER1_RERUN_PLAN.md` to require `--data_dir /data_x/aa007878/galileo/data_tier1_6` for Tier‑1 and to run all preflights/validators via `conda run -n galileo`.
- Updated `docs/paper/STATUS.md` to record the 6-benchmark Tier‑1 SSOT decision.

### 2026-02-24 20:40 KST — Run reliability: force live logging for tmux Tier‑1 runs

- Updated `docs/paper/NEXT_TIER1_RERUN_PLAN.md` to use `stdbuf -oL -eL` + `python -u` so `run.log` updates live (avoids silent-buffer confusion).

### 2026-02-24 21:09 KST — Experiments: Tier‑1(6) Phi‑3‑mini seed1 pilot is progressing (silent log)

- Running: `results/tier1_phi3mini_seed1_tier1_6_pilot_200_512_20260224_202211/` (GPU4).
- Output evidence (log may be buffered): JSONLs created for ARC-Easy + TriviaQA (initial/adversarial/recovery).

### 2026-02-24 21:49 KST — Runbook: add JSONL-mtime stall watchdog for silent vLLM runs

- Updated `docs/paper/NEXT_TIER1_RERUN_PLAN.md` with a 10-minute JSONL mtime watchdog rule (kill tmux if no output changes) to prevent long GPU burns when `run.log` is buffered.

### 2026-02-24 22:05 KST — CLI: fix --data_dir + --data_file precedence for single-file diagnostics

- `run_experiment.py`: when both `--data_dir` and `--data_file` are set, now restricts to that one JSONL under `data_dir` (previously `--data_dir` overrode and ran all JSONLs).

### 2026-02-24 23:32 KST — Runbook fix: paper_export requires --model_dir

- Updated `docs/paper/NEXT_TIER1_RERUN_PLAN.md` to pass `--model_dir $OUT/${MODEL##*/}` to `scripts/paper_export.py` (it is required).

### 2026-02-25 02:30 KST — Table1: regenerate artifact from results_paper exports (Nemo Tier1(6) pilots)

- Generated `docs/paper/artifacts/table1_from_results_paper_exports_20260225_0229.csv` from nlp8 `results_paper/`.
- Includes Mistral‑Nemo seed1–2 Tier‑1(6-benchmark) pilots with Recovery@flip.

### 2026-02-25 04:02 KST — Tooling: add JSONL-mtime-based run liveness checker

- Added `scripts/check_run_progress.py` to quickly decide alive vs stalled when `run.log` is silent.
- Signals completion via presence of root CSVs: `initial_accuracy.csv`, `adversarial_survival.csv`, `recovery_accuracy.csv`.

### 2026-02-25 04:10 KST — Tooling: add run progress checker (JSONL mtime + root CSV)

- Added `scripts/check_run_progress.py` to quickly detect silent stalls via JSONL mtimes and root CSV completion.

### 2026-02-27 17:28 KST — Paper: disambiguate time-to-first metric acronyms in Related Work

- Added an explicit Related Work sentence noting `ToF` acronym collisions and standardizing our terminology on `TTF` (time-to-first-failure) + explicit `NoF` definition pointer (\Sef{sec:metrics}).

### 2026-03-11 00:32 KST — Experiments: confirmed grounded Qwen7B multiseed result root + validation artifacts

- Confirmed the grounded multiseed queue command in `tmp/results/queue_grounded_multiseed_20260310_232647.sh` launches `scripts/run_qwen7b_multiseed_single_gpu_tmux.sh` with `RUN_GROUP=qwen7b_grounded_multiseed_gpu5`, seeds `1,2,3`, GPU `5`, and grounded GSM8K/ARC dataset roots.
- Confirmed result root exists at `tmp/results/qwen7b_grounded_multiseed_gpu5_20260310_232747/`.
- Verified `GLOBAL_VALIDATE.log` reports `[OK]` for ARC + GSM8K `seed_1`–`seed_3` `paper_exports` and `[OK] runner_metadata parity`.
- Existing extracted artifacts remain:
  - `docs/paper/artifacts/qwen7b_grounded_multiseed_seed1-3_metrics_20260310.csv`
  - `docs/paper/artifacts/qwen7b_grounded_multiseed_seed1-3_deltas_20260310.csv`
- Note: `scripts/check_run_progress.py` warns on this root because it expects root-level CSV summaries, but that does not invalidate the multiseed per-seed export layout validated above.

### 2026-03-11 00:41 KST — Docs/evidence: refreshed claim→evidence map for March 10 Qwen7B multiseed package

- Rechecked that the three March 10, 2026 Qwen7B multiseed roots still validate:
  - `tmp/results/qwen7b_evidence_multiseed_gpu5_20260310_231212/`
  - `tmp/results/qwen7b_grounded_multiseed_gpu5_20260310_232747/`
  - `tmp/results/qwen7b_evidencegate_multiseed_gpu5_20260310_234316/`
- Confirmed the tracked multiseed metric/delta artifacts and three-way comparison artifact still exist under `docs/paper/artifacts/`.
- Updated `docs/paper/CLAIM_EVIDENCE_MAP.md` to add:
  - an explicit March 10 Qwen7B multiseed proof bundle section,
  - new claim-level proof pointers for same-model multiseed robustness and evidence-gate trade-off framing,
  - the concrete regeneration entry point `scripts/aggregate_condition_multiseed.py` for multiseed metric/delta aggregation.
- No GPU rerun was needed because equivalent validated outputs already existed.

### 2026-03-11 00:46 KST — Cleanup: removed redundant lowercase claim-map symlink

- Audited `docs/paper/claim_evidence_map.md` versus `docs/paper/CLAIM_EVIDENCE_MAP.md`.
- Confirmed the lowercase path was only a symlink to the uppercase SSOT file and found no active non-archive references to the lowercase path in current README/docs/scripts.
- Removed the redundant lowercase symlink to reduce duplicate-path drift risk; the uppercase `docs/paper/CLAIM_EVIDENCE_MAP.md` remains the only tracked claim-map SSOT.

### 2026-03-11 00:55 KST — Paper evidence: added strong-vs-weak claim ledger for the March 10 package

- Revalidated the weaker non-multiseed roots I might cite as directional-only context:
  - `tmp/results/smoke_gpu5_20260310_184715/`
  - `tmp/results/pilot50_gpu5_20260310_185825/`
  - `tmp/results/main_arc_gpu6_20260310_191906/`
- Reused the already-validated Qwen7B multiseed roots for evidence / grounded / evidence-gate and extracted stable Survival@5 proof points from the tracked CSV artifacts.
- Added `docs/paper/PAPER_EVIDENCE_STATUS_20260310.md` to separate:
  - reproducible supported claims,
  - promising but insufficient observations,
  - missing-evidence gaps,
  - safe body-ready wording candidates.
- Added a pointer to that new summary in `docs/paper/CLAIM_EVIDENCE_MAP.md`.
- No paper-body edit was made; this cycle stays in the evidence/supporting-docs layer only.
