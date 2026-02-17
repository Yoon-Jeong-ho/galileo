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

Ground-truth tasks에서 multi-turn persona pressure 하에 **정답 유지(survival)**, **최초 붕괴 시점(TOF)**, **붕괴 후 회복(recovery)** 을 측정하고, **Neutral Re-asking Control**로 drift를 분리하는 평가 프로토콜/벤치마크.

---

## 1) NOW (what we are doing / what is true *right now*)

- We are in a **10-min heartbeat loop**; each heartbeat must deliver **one primary lane** result and keep continuity via `STATUS.md` + `HEARTBEAT_LOG.md`.
- **Remote experiments policy:** use **nlp8**; GPUs **4/5/6** only; `tmux` required; paper-ready runs must include `paper_exports/*` + `metadata.json` + `runner_metadata.json` + validator `[OK]` + parity.
- **Canonical remote launcher (anti-drift):** prefer `scripts/run_multiseed_tmux.sh` (streams logs + writes `runner_metadata.json`). Other launch scripts are allowed only when explicitly justified in `results/<run>/run.log`.
- **Validator health (paper SSOT):** `results_paper/` global validation + runner-metadata parity is **[OK]** (Tier‑1 families incl. Llama‑3.2‑3B, Phi‑3‑mini, **Mistral‑Nemo**). Incomplete runs are quarantined under `results_paper_incomplete/` to keep paper SSOT clean.
- **Phi-3-mini Tier-1 (cross-family):** seed1–2 are **paper-ready** (validated) under `results_paper/tier1_phi3mini_seed{1,2}_20260217_*`.
- **Mistral-Nemo Tier-1 (cross-family):** seed1–2 are **paper-ready** under:
  - `results_paper/tier1_mistralnemo_seed1_20260217_173907/`
  - `results_paper/tier1_mistralnemo_seed2_20260217_180951/`
- **Decoding sweep (seed1–2) done:** `results_paper/qwen_temp0_seed{1,2}` + `results_paper/qwen_temp0p7_seed{1,2}` are paper-ready; `results_paper/GLOBAL_VALIDATE.log` remains all `[OK]`.
- **Current risk:** process drift (conflicting docs about host/GPU policy, lane starvation, missing commits) → we keep SSOT docs aligned (STATUS/CHECKLIST/RUNBOOK).
- **Cross-family extension note:** attempted adding Gemma2 (google/gemma-2-2b-it) on nlp8 RTX8000 via vLLM; it fails due to (i) max_model_len>8192 guardrail and (ii) Triton unified-attention shared-memory OOR on cc7.5. Avoid Gemma2 on this hardware unless we change vLLM backend/settings.
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
- ✅ Generated **submission-ready SVG figures** from tracked CSV artifacts under `docs/paper/figures/` (seed1–4; survival curves + ΔSurvival@5 + ΔFail@1 + ΔRecovery + Table W effect deltas).
- ✅ Tightened the reviewer-facing **Claims → evidence** table to include explicit **LaTeX figure labels** (reduces proof-pointer drift): `docs/paper/PAPER_DRAFT_EN.md` (§9).
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
- ✅ Canonical cross-family figure filename (anti-drift): `docs/paper/figures/cross_family_survival_r5_control_vs_logicaltrap_seed1-2_20260217.svg`
- ✅ Metric definitions SSOT: `docs/paper/FIGURE_CAPTIONS.md` (Survival@r / Flip / TOF)

---

## 3) TOP GAPS (what still blocks paper quality)

1) **LaTeX build readiness (PDF figures):** ✅ PDFs can now be generated **without sudo** via Inkscape AppImage (`scripts/get_inkscape_appimage.sh` → `scripts/convert_figures_svg_to_pdf.sh`; output `paper_figures/pdf/*.pdf`). ✅ LaTeX smoke-tests compile in CI: (i) generic skeleton `docs/paper/latex_skeleton/main.tex`, and (ii) EMNLP2023 template skeleton `docs/paper/latex_skeleton_emnlp2023/main_emnlp2023.tex` (both built by `latex-smoketest` and uploaded as artifacts). Remaining step: switch/confirm the **current EMNLP year** template (2024/2025) if required by submission instructions.
2) **Claim→evidence map completion:** ✅ claim-map skeleton now includes an Abstract/Intro checklist + corrected figure/artifact paths; remaining work is to (i) tie each Abstract/Intro sentence to a specific checklist item and (ii) keep it in sync as the draft changes.
   - SSOT: `docs/paper/CLAIM_EVIDENCE_MAP.md`
3) **Experiment extension decision (Tier‑1 only):** decide whether the next marginal compute should go to (a) decoding sensitivity sweep vs (b) an additional model family vs (c) more seeds (only if CI looks fragile).

**Update:** Llama‑3.2‑3B‑Instruct, Phi‑3‑mini, and **Mistral‑Nemo** all have Tier‑1 **seeds 1–2** that are paper-ready and reflected in the tracked cross-family artifact/figure set (see `docs/paper/artifacts/tier1_*_survival_summary_*.csv` and the canonical SVG under `docs/paper/figures/`).

---

## 4) NEXT HEARTBEAT (ONE step)

**Paper writing: produce a “수정/보완해야 할 것만” TODO list in KO draft (reviewer-risk only).**

- Deliverable: `docs/paper/PAPER_DRAFT_KO.md`에 섹션 0 또는 별도 섹션으로, 각 항목을
  - “현재 ~~가 부족/문제이고 → ~~해서 → ~~로 수정해야 한다”
  형태로 8–12개만 정리 (불필요한 일반론/칭찬 없이, 수정 포인트만).

---

## 5) Notes / constraints

- Keep runs light (avoid CPU overload; one heavy run at a time).
- If any repo file changes: commit+push (no carry-over ambiguity).
