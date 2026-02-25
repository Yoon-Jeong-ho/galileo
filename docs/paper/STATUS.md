# GALILEO EMNLP Main — STATUS (rolling)

> 이 파일은 **매 heartbeat마다 갱신(리뉴얼)** 하는 단 하나의 상태판입니다.
> - "지금 현재" 우리가 어디까지 왔는지
> - 남은 핵심 갭이 무엇인지
> - 다음 1-step이 무엇인지
>
> 규칙:
> 1) heartbeat 시작 시 이 파일을 먼저 읽는다.
> 2) heartbeat 종료 시 이 파일을 업데이트한다(중복 문구 금지, 최신 상태로 rewrite).
> 3) 상세 타임라인은 `docs/paper/HEARTBEAT_LOG.md`(append-only)에 남긴다.

---

## 0) One-line pitch (draft)

Ground-truth tasks에서 multi-turn persona pressure 하에 **정답 유지(survival)**, **최초 실패 시점(TTF; time-to-first-failure)**, **붕괴 후 회복(recovery)** 을 측정하고, **NRC(Neutral Re-asking Control)**로 drift를 분리하는 평가 프로토콜/벤치마크.

---

## 1) NOW (what we are doing / what is true *right now*)

- We are in a **10-min heartbeat loop**; each heartbeat must deliver **one primary lane** result and keep continuity via `STATUS.md` + `HEARTBEAT_LOG.md`.
- **Active run (nlp8, Tier‑1 6-benchmark pilot):** Phi‑3‑mini seed1 is running on GPU4 (tmux session created at 20:22 KST):
  - `results/tier1_phi3mini_seed1_tier1_6_pilot_200_512_20260224_202211/`
  - Note: `run.log` is currently silent due to buffering; monitor progress via output JSONLs under `Phi-3-mini-4k-instruct/` (size/mtime).
- **Remote experiments SSOT (confirmed):** paper-ready SSOT is **nlp8** (repo `/data_x/aa007878/galileo`, GPUs **0–6**, but **ONLY** those not used by other users). Heartbeat banner references to `nlp16` are stale.
- **Tier‑1 benchmark SSOT (2026-02-24 decision):** all Tier‑1 runs must use the full 6-task set via `--data_dir /data_x/aa007878/galileo/data_tier1_6` (do not rely on defaults).
- **Remote sanity (2026-02-22 12:09 KST):** GPU3 is currently occupied by **aa007878** (`Training/train_llama.py` for PrivacyRestore; ~21.6GiB). GPUs 1/2/4/5/6 appear idle; GPU0 looks idle in `nvidia-smi` but **CUDA alloc preflight FAIL** → exclude GPU0 for launches until it recovers. Avoid launching GALILEO on GPU3 until that job finishes.
- **Idle-GPU usage clarification (2026-02-20):** when GPUs 0/1 are idle on nlp8, they are valid launch targets under SSOT policy (0–6 idle-only). We should not wait for 4–6-only windows because the heartbeat banner text is stale.
- **GPU occupancy check fix (2026-02-20):** `nvidia-smi --query-compute-apps` does **not** support a `username` field on this host; use PID→user mapping via `ps` if needed.
  - **Observed now (nlp16):** GPUs 4/5 are heavily contended (e.g., `jslee-fusion-distill-vllm-v1` holding ~47GiB on GPU5; `eval-worker-gpu1` ~37GiB on GPU4), causing vLLM EngineCore init failures for new TP=4 starts.
  - **Action:** do not blindly relaunch vLLM when free-VRAM is low; fingerprint PIDs first.
- **Canonical remote launcher (anti-drift):** prefer `scripts/run_multiseed_tmux.sh` (streams logs + writes `runner_metadata.json`). Other launch scripts are allowed only when explicitly justified in `results/<run>/run.log`.
- **Validator health (paper SSOT):** `results_paper/` global validation + runner-metadata parity is **[OK]** (Tier‑1 families incl. Llama‑3.2‑3B, Phi‑3‑mini, **Phi‑3.5‑mini**, **Mistral‑Nemo**, **DeepSeek‑LLM‑7B‑Chat**). Incomplete runs are quarantined under `results_paper_incomplete/` to keep paper SSOT clean.
- **Phi-3-mini Tier-1 (cross-family):** seed1–2 are **paper-ready** (validated) under `results_paper/tier1_phi3mini_seed{1,2}_20260217_*`.
- **Phi-3.5-mini Tier-1 (cross-family):** seed1–2 are **paper-ready** (validated) and staged under:
  - `results_paper/tier1_phi35mini_seed1_20260219_143555/`
  - `results_paper/tier1_phi35mini_seed2_20260219_143555/`
  - Note: these runs show near-total collapse (Survival@5=0) and `max_tokens` was capped to **1** due to the batch-wise cap logic (`prompt_tokens + max_tokens + reserve_tokens <= max_model_len`, with `reserve_tokens=256` in `inference.py`). **Do not use Phi‑3.5‑mini as a headline cross-family evidence point until rerun under settings where token-capping is absent (or explicitly analyzed).** Keep it as a “stress-test/settings sensitivity” datapoint (appendix-only) unless/until we rerun with more headroom (e.g., smaller R, smaller reserve, or a larger max context).
- **Mistral-Nemo Tier-1 (cross-family):** seed1–2 are **paper-ready** under:
  - `results_paper/tier1_mistralnemo_seed1_20260217_173907/`
  - `results_paper/tier1_mistralnemo_seed2_20260217_180951/`
- **DeepSeek-LLM-7B-Chat Tier-1 (cross-family):** seed1–2 are **paper-ready** and staged into `results_paper/`:
  - `results_paper/tier1_deepseek7b_seed1_20260221_052948/`
  - `results_paper/tier1_deepseek7b_seed2_20260221_061004/`
  - Artifact (CSV): `docs/paper/artifacts/tier1_deepseek7b_seed1-2_survival_summary_20260221.csv`
- **Yi-6B-Chat Tier-1 (cross-family):** seed1–2 are **paper-ready** and staged into `results_paper/`:
  - `results_paper/tier1_yi6b_seed1_20260221_122636/`
  - `results_paper/tier1_yi6b_seed2_20260221_125813/`
  - Artifact (CSV): `docs/paper/artifacts/tier1_yi6b_seed1-2_survival_summary_20260221.csv`
- **StableLM-2-1.6B chat Tier-1 attempt (2026-02-21):** **aborted / not citable** due to repeated vLLM batch cap warnings that force generations down to 1 token (e.g., `requested max_tokens=256 capped to 1`). This indicates a context/packing feasibility mismatch for our R=5 prompts at `max_model_len=4096` on this stack; do **not** stage into `results_paper/`.
- **Canonical cross-family figure (SVG):** `docs/paper/figures/cross_family_survival_r5_control_vs_logicaltrap_seed1-2_20260221.svg`
- **Decoding sweep (seed1–2) done:** `results_paper/qwen_temp0_seed{1,2}` + `results_paper/qwen_temp0p7_seed{1,2}` are paper-ready; `results_paper/GLOBAL_VALIDATE.log` remains all `[OK]`.
- **Tier‑1 Qwen2.5‑14B‑Instruct (cross-family extension):**
  - seed1 is paper-ready and staged into `results_paper/`.
  - seed2 is **paper-ready** and staged into `results_paper/`:
    - run: `results/tier1_qwen2p5_14b_seed2_20260219_053824/`
    - staged: `results_paper/tier1_qwen2p5_14b_seed2_20260219_053824/paper_exports`
    - validator: `[OK] .../paper_exports` and global `[OK] runner_metadata parity`
- **Current risk:** process drift (conflicting docs about host/GPU policy, lane starvation, missing commits) → we keep SSOT docs aligned (STATUS/CHECKLIST/RUNBOOK).
- **Paper clarity micro-risk:** repeated long phrase “Neutral Re-asking Control” can add cognitive load; prefer introducing the acronym **NRC** once (Abstract) and using it consistently thereafter.
- **LaTeX SSOT hygiene (2026-02-20):** Appendix prompt/command blocks now use `fvextra` line-breaking `Verbatim`, eliminating noisy `Overfull \\hbox` warnings from long lines (better PDF-first iteration).
- **LaTeX Metrics section hygiene (2026-02-20):** rewrote long hazard/TOF expressions to avoid overfull math boxes; current `main.log` has no `Overfull \\hbox` entries.
- **Main-text space management (2026-02-20):** moved Recovery persona-wise plot + decoding sweep figure to Appendix (`Additional results`), keeping main Results focused on Survival/Fail@1/Cross-family.
- **Page-counting hygiene (2026-02-20):** LaTeX SSOT now supports a build-time `\CAMERAREADY` switch (no file edits) to compile with/without EMNLP review mode (line numbers). Use camera-ready mode for closer-to-submission page budgeting.
- **Limitations-excluded budgeting (2026-02-20):** added `scripts/report_latex_page_budget.sh` + LaTeX log markers to compute "main pages excluding Limitations" automatically in the PDF-first SSOT.
- **Structure hygiene (2026-02-20):** split `Discussion` and `Limitations` into separate \section blocks so the Limitations-excluded page budget is unambiguous.
- **Main-text expansion tracking (2026-02-20):** after Intro + Protocol notation + Results narrative expansion (Survival/Fail@1/Recovery/Cross-family) + Table~\ref{tab:main} refactor, camera-ready budget is now main(excl Limitations)=4 pages (target: 8).
- **Cross-family extension note (Gemma2):** attempted adding Gemma2 (google/gemma-2-2b-it) on nlp8 RTX8000 via vLLM; it fails due to (i) max_model_len>8192 guardrail and (ii) Triton unified-attention shared-memory OOR on cc7.5. Avoid Gemma2 on this hardware unless we change vLLM backend/settings.
- **Cross-family extension note (Falcon-7B):** `tiiuae/falcon-7b-instruct` fails at vLLM init (`FalconConfig` missing `rope_parameters`) → likely transformers/vLLM compatibility issue; run is **incomplete** (no exports).
- **Cross-family extension note (Pythia-2.8B):** fails because we requested `--max_model_len 4096` but model-derived max is 2048; fix by using `--max_model_len 2048` (preferred) instead of `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`.
- **Pythia Tier-1 decision (2026-02-19 late):** `tier1_pythia2p8b_seed2_20260219_211411` was terminated after repeated nonterminal loops (800/800 then restart phases, no `EXPERIMENT COMPLETE`, no `paper_exports/`). Treat Pythia seed1/seed2 as **non-citable on current stack**; no further blind retries.
- **Current infra blocker (2026-02-20 early):** even when GPU0 looked idle by `nvidia-smi`, direct CUDA preflight (`CUDA_VISIBLE_DEVICES=0` + torch tensor alloc) failed with `cudaErrorDevicesUnavailable`; this indicates launch-time device availability races beyond simple idle snapshots.
- **Runbook fix (2026-02-20):** the runbook previously referenced a non-existent `scripts/check_cuda_preflight.py`; it now uses a pure-torch alloc sanity snippet + the existing `scripts/preflight_vllm_model.py` for model init.
- **Launch gating (2026-02-20):** treat “GPU is idle” as **necessary but not sufficient**; require torch alloc preflight `OK cuda alloc` on the chosen GPU before launching any Tier‑1 run.
- **Cross-family extension note (Zephyr-7B):** older run `tier1_zephyr7b_seed1_20260217_150053` is incomplete (empty `run.log`), but **Zephyr‑7B seeds 1–2 are now paper-ready** and integrated into the canonical cross-family artifacts/figure set (`..._20260218`).
- **EXAONE Tier-1 attempt status:** currently **failed/incomplete** due to `ImportError: RopeParameters` from `transformers.modeling_rope_utils` when loading EXAONE remote code; kept under `results_paper_incomplete/` (do not cite as evidence).

---

## 2) RECENTLY DONE (verifiable, high-signal)

### Qualitative / taxonomy (interpretability)
- ✅ KO qualitative taxonomy substantially expanded and reorganized with multi-seed snapshot tables + multi-seed representative examples (`docs/paper/PAPER_DRAFT_KO.md`).
- ✅ EN Appendix A.2 added: evaluator-dependent flip taxonomy (boundary/partial/semantic) + explicit separation of rare format/extraction artifacts (`docs/paper/PAPER_DRAFT_EN.md`).

### Experiments / artifacts
- ✅ **Seed1–4 (control vs persona) are auditable green** (Qwen2.5-7B-Instruct; 80 samples/seed) with validated `paper_exports/`.
- ✅ **Mistral-7B seeds 1–2** are auditable green and included in `results_paper/`.
- ✅ **Llama-3.1-8B-Instruct seeds 1–2** are auditable green and included in `results_paper/`.
- ✅ **Tier‑1 cross-family extension:** `meta-llama/Llama-3.2-3B-Instruct` **seeds 1–2** are now auditable green and staged under:
  - `results_paper/tier1_llama3_3b_seed1_20260212_030426/`
  - `results_paper/tier1_llama3_3b_seed2_20260212_042339/`.
- ✅ **Tier‑1 cross-family extension:** `microsoft/Phi-3-mini-4k-instruct` **seeds 1–2** are auditable green (paper-ready):
  - `results_paper/tier1_phi3mini_seed1_20260217_011737/`
  - `results_paper/tier1_phi3mini_seed2_20260217_033953/`
- ✅ **Recovery-variant ablation (verify_then_answer; Qwen seeds 1–2)** is auditable green and included in `results_paper/` (aliases `qwen_vta_seed1`, `qwen_vta_seed2`).
- ✅ Introduced a **paper-only validation root** `results_paper/` on nlp8 to keep global validation stable for cited runs (parity PASS).
- ✅ **Table W artifacts tracked** under `docs/paper/artifacts/` and draft AUTO block updated to seed1–4.

### Paper writing / positioning
- ✅ Standardized and *de-duplicated* the acronym **NRC** for “Neutral Re-asking Control” in the EN draft (keep first expansion; thereafter use NRC only) to reduce repetition and make persona–control comparisons easier to parse: `docs/paper/PAPER_DRAFT_EN.md`.
- ✅ Related Work §6.4 now explicitly frames NRC as the missing “evidence-free, matched-length” counterfactual that separates pressure mechanisms from generic drift / evaluation framing confounds (positioning vs TRUTH DECAY / Challenging the Evaluator), with an explicit proof-pointer to Protocol+TableW. Added the same kind of proof-pointer for the ReviseQA contrast (no-new-evidence vs belief revision): `docs/paper/PAPER_DRAFT_EN.md`.
- ✅ Abstract headline findings now end with explicit, minimal proof-pointers (Table W + key figs), reducing reviewer search cost. The quantified Table~W deltas in the Abstract are explicitly marked as persona-weighted aggregates: `docs/paper/PAPER_DRAFT_EN.md`.
- ✅ C3 intervention ablation paragraph now contains an explicit “proof pointer” to the baseline-vs-variant recovery-gap comparison artifact (reduces reviewer hunting): `docs/paper/PAPER_DRAFT_EN.md`.
- ✅ Intro now has a compact section-pointer sentence tying ground-truth tasks (§2) + NRC + protocol figure to reduce reviewer navigation overhead, plus an explicit pointer from the comparability caveat to the protocol/reporting-modes spec (§3): `docs/paper/PAPER_DRAFT_EN.md`.
- ✅ KO draft now includes an in-place **“수정/보완해야 할 것만” 리비전 TODO (SSOT 발췌, 8–12개)** for reviewer-risk-only edits: `docs/paper/PAPER_DRAFT_KO.md`.
- ✅ Generated **submission-ready SVG figures** from tracked CSV artifacts under `docs/paper/figures/` (seed1–4; survival curves + ΔSurvival@5 + ΔFail@1 + ΔRecovery + Table W effect deltas).
- ✅ Tightened the reviewer-facing **Claims → evidence** table to include explicit **LaTeX figure labels** (reduces proof-pointer drift): `docs/paper/PAPER_DRAFT_EN.md` (§9).
- ✅ Synced `docs/paper/CLAIM_EVIDENCE_MAP.md` cross-family evidence pointers with current SSOT artifacts (removed stale Zephyr pending placeholder; now points to `tier1_zephyr7b_seed1-2_survival_summary_20260218.csv`).
- ✅ Refreshed Tier-1 gap ledger to explicitly quarantine recent Pythia failures (`seed1_20260219_*`, `seed2_20260219_211411`) and codified no-blind-retry rule in `docs/paper/TIER1_GAP_CHECKLIST_20260219.md`.
- ✅ Results section now has artifact-cited prose for survival/TOF/recovery/control-comparison (seed1–4) + a Results preface stating the seed/"auditable green" convention.
- ✅ Added an explicit **Abstract proof-pointer line** (protocol/survival/TOF/recovery/Table W) to reduce reviewer search cost (`docs/paper/PAPER_DRAFT_EN.md`).
- ✅ Made Intro §1.1 reviewer-skim friendly by adding an **“Evidence at a glance”** bullet list with explicit figure/table pointers; kept SSOT alignment via `docs/paper/CLAIM_EVIDENCE_MAP.md`.
- ✅ Table W (control vs persona) is supported by both mean±std and Δ(effect-size) tracked artifacts, and the Results text cites them.
- ✅ De-duplicated “no-new-evidence” framing between protocol/control bullets and ReviseQA positioning (`docs/paper/PAPER_DRAFT_EN.md`).
- ✅ Related-work tightening landed for:
  - TRUTH DECAY (protocol + models/datasets)
  - Challenging the Evaluator (protocol + accept-rate framing)
  - Draft positioning sentences updated accordingly.

### Process guardrails
- ✅ Added SSOT heartbeat prompt: `docs/paper/HEARTBEAT_PROMPT.md`
- ✅ Added heartbeat checklist guardrails: `docs/paper/HEARTBEAT_CHECKLIST.md`
- ✅ De-confused deprecated `nlp16` SSH note to reduce copy/paste drift: `docs/paper/SSH_TROUBLESHOOT_NLP16.md`
- ✅ Canonical cross-family figure filename (anti-drift): `docs/paper/figures/cross_family_survival_r5_control_vs_logicaltrap_seed1-2_20260221.svg`
- ✅ Metric definitions SSOT: `docs/paper/FIGURE_CAPTIONS.md` (Survival@r / Flip / TOF)
- ✅ Added CUDA preflight helper + runbook gate to prevent false-idle launch failures:
  - `scripts/check_cuda_preflight.py`
  - `docs/paper/REMOTE_EXPERIMENTS_RUNBOOK.md`

---

## 3) TOP GAPS (what still blocks paper quality)

1) **LaTeX build readiness (PDF figures):** ✅ PDFs can now be generated **without sudo** via Inkscape AppImage (`scripts/get_inkscape_appimage.sh` → `scripts/convert_figures_svg_to_pdf.sh`; output `paper_figures/pdf/*.pdf`). ✅ LaTeX smoke-tests compile in CI: (i) generic skeleton `docs/paper/latex_skeleton/main.tex`, and (ii) EMNLP2023 template skeleton `docs/paper/latex_skeleton_emnlp2023/main_emnlp2023.tex` (both built by `latex-smoketest` and uploaded as artifacts). **Local note:** this environment currently lacks TeX (`latexmk/pdflatex` not installed), so run LaTeX smoke-tests in CI or a TeX-enabled machine. Remaining step: switch/confirm the **current EMNLP year** template (2024/2025) if required by submission instructions.
   - Canonical baseline per ACLPUB formatting guidelines: `acl-org/acl-style-files` (we pin a specific snapshot SHA for reproducibility; helper: `bash scripts/get_acl_style_files.sh <ref>`).
2) **Claim→evidence map completion:** ✅ claim-map skeleton now includes an Abstract/Intro checklist + corrected figure/artifact paths; remaining work is to (i) tie each Abstract/Intro sentence to a specific checklist item and (ii) keep it in sync as the draft changes. ✅ Added a short “Interpretation guardrails” paragraph right before Results to prevent C vs C_p / Table-W pooling confusion.
   - SSOT: `docs/paper/CLAIM_EVIDENCE_MAP.md`
3) **Experiment extension decision (Tier‑1 only):** decide whether the next marginal compute should go to (a) decoding sensitivity sweep vs (b) an additional model family vs (c) more seeds (only if CI looks fragile).
   - **Update (2026-02-18):** Zephyr‑7B Tier‑1 **seeds 1–2 are now paper-ready**; cross-family integration can proceed (summary CSV + figure regen).

**Update:** Llama‑3.2‑3B‑Instruct, Phi‑3‑mini, and **Mistral‑Nemo** all have Tier‑1 **seeds 1–2** that are paper-ready and reflected in the tracked cross-family artifact/figure set (see `docs/paper/artifacts/tier1_*_survival_summary_*.csv` and the canonical SVG under `docs/paper/figures/`).

**Update (2026-02-18):** Zephyr‑7B Tier‑1 **seeds 1–2 are paper-ready** and already integrated into the canonical cross-family figure + artifacts (`..._20260218`).

**Update (2026-02-18 evening):** nlp16 is reachable and contains legacy `results/` runs (e.g., `results/rerun_persona_seed1_20260210_1140/`), but we should treat it as **legacy/non-SSOT**: the specific run we tailed is **not progressing** (no live `run_experiment.py`, log mtime stale). New vLLM starts are also unreliable due to heavy external GPU occupancy on 4–7.

**SSOT clarification:** all auditable “paper-ready” experiment work is SSOT on **nlp8** (repo `/data_x/aa007878/galileo`, GPUs **0–6 idle-only**) per `docs/paper/REMOTE_EXPERIMENTS_RUNBOOK.md`.

---

## 4) NEXT HEARTBEAT (ONE step)

**Dev/experiments lane (Table 1): finish Recovery@flip auto-fill end-to-end on SSOT (nlp8).**

- Done (code): `paper_export.py` can now export normalized `paper_exports/recovery_accuracy.csv` (NRC id stable).
- Done (code): `make_table1_from_results_paper_exports.py` computes `nrc_recovery/persona_recovery/delta_recovery` when recovery export exists.
- Done (code): `gen_latex_table1_from_artifacts.py` ingests Recovery from the same `table1_from_results_paper_exports_*.csv` artifact (no alias hardcoding).

**Next action (SSOT, nlp8):** backfill recovery exports for *one* missing Tier‑1 family beyond Mistral.

- Triage (2026-02-24): on nlp8, `results/` currently contains Phi‑3.5‑mini runs, but **no** `*phi3mini*` or `*mistralnemo*` run dirs were found under `results/` (maxdepth=1), and no `phi|nemo` aliases exist under `results_paper/`.

So the immediate next step becomes:
1) decide target family: **Phi‑3‑mini** or **Mistral‑Nemo**
2) either (a) recover the run dirs (if archived elsewhere) **or** (b) rerun seed1–2 under standardized Tier‑1 to recreate those paper-ready runs
3) once run dirs exist, restage into `results_paper/` + run `paper_export.py` (to include recovery) + regenerate `docs/paper/artifacts/table1_from_results_paper_exports_<today>.csv`

---

## 5) Roadmap to deadline (2026-02-28)

- SSOT roadmap file: `docs/paper/ROADMAP_TO_20260228.md`
- This is the execution anchor for heartbeat prioritization until submission lock.

## 6) Notes / constraints

- Keep runs light (avoid CPU overload; one heavy run at a time).
- If any repo file changes: commit+push (no carry-over ambiguity).

- (2026-02-25) Table‑1 artifact updated from `results_paper/` exports to include **Mistral‑Nemo Tier‑1(6-benchmark) pilots seed1–2** with Recovery@flip populated:
  - `docs/paper/artifacts/table1_from_results_paper_exports_20260225_0229.csv`

- (2026-02-25) Added a stdlib-only liveness/completion checker for remote runs (mitigates silent `run.log` buffering + silent-stall GPU burn):
  - `scripts/check_run_progress.py` (checks JSONL mtimes + root CSV completion)

- (2026-02-25) Fixed `scripts/preflight_vllm_model.py` default behavior: preflight now caps `max_model_len` to 4096 by default to avoid KV-cache init failures on long-context models (e.g., Nemo 128k) on RTX8000.

- (2026-02-25) `preflight_vllm_model.py`: added `--enforce_eager` to avoid heavy CUDA graph capture/torch.compile during preflight (helps prevent SIGKILL/timeouts on shared GPUs).

- (2026-02-25) Added robustness flag to mitigate scale-up stalls: `run_experiment.py --reset_engine_between_phases` (recreates vLLM engine between Phase 1/2/3).

- (2026-02-25) Added stronger robustness flag: `run_experiment.py --reset_engine_between_tasks` (recreates vLLM engine per dataset JSONL) to mitigate scale-up stalls that survive phase-level resets.

- (2026-02-25) Added `scripts/merge_results_csvs.py` to merge root CSV summaries across multiple partial runs (enables dataset-by-dataset isolation when long scale-up sweeps stall).

- (2026-02-25) Mitigation for scale-up stalls: vLLM generation is now chunked to avoid huge batches. Control via env `GALILEO_MAX_BATCH_SIZE` (default 64). See `inference.py`.

- (2026-02-25) Exposed vLLM stability knobs for long scale-up runs: `run_experiment.py --gpu_memory_utilization <float>` and `--enforce_eager` (plumbs into `InferenceEngine`/vLLM).
- (2026-02-25) Nemo scale-up instability update: multiple Nemo runs stall at/after `gsm8k_initial.jsonl` (no JSONL mtime updates; root CSVs absent). For diagnosing silent stalls, prefer `scripts/check_run_progress.py` (JSONL mtimes + root CSV presence) over `run.log`.
- (2026-02-25) Debugging note: Llama-3.2-3B smoke runs emit `[dbg ...]` logs and complete, but Nemo smoke runs can produce **no stdout/stderr at all** (even with import breadcrumbs), while leaving a `VLLM::EngineCore` process on GPU. Treat as an early/hard hang and kill quickly to avoid GPU burn.
