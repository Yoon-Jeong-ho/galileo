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
- **Validator health (paper SSOT):** `results_paper/GLOBAL_VALIDATE.log` is **all [OK]** (includes `qwen_vta_seed1/2`).
- **Decoding sweep (seed1–2) done:** `results_paper/qwen_temp0_seed{1,2}` + `results_paper/qwen_temp0p7_seed{1,2}` are paper-ready; `results_paper/GLOBAL_VALIDATE.log` remains all `[OK]`.
- **Current risk:** process drift (conflicting docs about host/GPU policy, lane starvation, missing commits) → we keep SSOT docs aligned (STATUS/CHECKLIST/RUNBOOK).
- **Cross-family extension note:** attempted adding Gemma2 (google/gemma-2-2b-it) on nlp8 RTX8000 via vLLM; it fails due to (i) max_model_len>8192 guardrail and (ii) Triton unified-attention shared-memory OOR on cc7.5. Avoid Gemma2 on this hardware unless we change vLLM backend/settings.

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
- ✅ **Recovery-variant ablation (verify_then_answer; Qwen seeds 1–2)** is auditable green and included in `results_paper/` (aliases `qwen_vta_seed1`, `qwen_vta_seed2`).
- ✅ Introduced a **paper-only validation root** `results_paper/` on nlp8 to keep global validation stable for cited runs (parity PASS).
- ✅ **Table W artifacts tracked** under `docs/paper/artifacts/` and draft AUTO block updated to seed1–4.

### Paper writing / positioning
- ✅ Generated **submission-ready SVG figures** from tracked CSV artifacts under `docs/paper/figures/` (seed1–4; survival curves + ΔSurvival@5 + ΔFail@1 + ΔRecovery + Table W effect deltas).
- ✅ Results section now has artifact-cited prose for survival/TOF/recovery/control-comparison (seed1–4) + a Results preface stating the seed/"auditable green" convention.
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

---

## 3) TOP GAPS (what still blocks paper quality)

1) **LaTeX build readiness (PDF figures):** ✅ PDFs can now be generated **without sudo** via Inkscape AppImage (`scripts/get_inkscape_appimage.sh` → `scripts/convert_figures_svg_to_pdf.sh`; output `paper_figures/pdf/*.pdf`). ✅ LaTeX smoke-tests compile in CI: (i) generic skeleton `docs/paper/latex_skeleton/main.tex`, and (ii) EMNLP2023 template skeleton `docs/paper/latex_skeleton_emnlp2023/main_emnlp2023.tex` (both built by `latex-smoketest` and uploaded as artifacts). Remaining step: switch/confirm the **current EMNLP year** template (2024/2025) if required by submission instructions.
2) **Claim→evidence map completion:** for each Abstract/Intro claim, pin 1 figure/table + 1 reproducer path (script + artifact/run alias) so reviewers can verify quickly.
   - SSOT: `docs/paper/CLAIM_EVIDENCE_MAP.md`
3) **Experiment extension decision (Tier‑1 only):** decide whether the next marginal compute should go to (a) decoding sensitivity sweep vs (b) an additional model family vs (c) more seeds (only if CI looks fragile).

**Update:** Llama‑3.2‑3B‑Instruct **seeds 1–2** are now paper-ready and staged under `results_paper/` (see `results_paper/GLOBAL_VALIDATE.log` for validator + parity `[OK]`).

---

## 4) NEXT HEARTBEAT (ONE step)

**Paper writing: integrate the qualitative taxonomy into the main narrative without inflating the paper.**

- Deliverable: update `docs/paper/PAPER_DRAFT_EN.md` to add 1–2 short cross-references from Results/Limitations to Appendix~A.2 (avoid new figures), and ensure terminology is consistent (boundary/partial/semantic + format/extraction failure).

---

## 5) Notes / constraints

- Keep runs light (avoid CPU overload; one heavy run at a time).
- If any repo file changes: commit+push (no carry-over ambiguity).
