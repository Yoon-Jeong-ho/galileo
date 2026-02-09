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
- **Remote experiments policy:** use **nlp8** only; temporary GPUs **4/5/6** only; `tmux` required; paper-ready runs must include `paper_exports/*` + `metadata.json` + `runner_metadata.json` + validator `[OK]` + parity.
- **Current risk:** process drift (wrong server prompt like nlp16, lane starvation, missing commits) → we are adding guardrails to eliminate it.

---

## 2) RECENTLY DONE (verifiable, high-signal)

### Experiments / artifacts
- ✅ **Seed1–4 (control vs persona) are auditable green** (Qwen2.5-7B-Instruct; 80 samples/seed) with validated `paper_exports/`.
- ✅ **Table W artifacts tracked** under `docs/paper/artifacts/` and draft AUTO block updated to seed1–4.

### Paper writing / positioning
- ✅ Results section now has artifact-cited prose for survival/TOF/recovery/control-comparison (seed1–4) + a Results preface stating the seed/"auditable green" convention.
- ✅ Table W (control vs persona) is supported by both mean±std and Δ(effect-size) tracked artifacts, and the Results text cites them.
- ✅ Related-work tightening landed for:
  - TRUTH DECAY (protocol + models/datasets)
  - Challenging the Evaluator (protocol + accept-rate framing)
  - Draft positioning sentences updated accordingly.

### Process guardrails
- ✅ Added SSOT heartbeat prompt: `docs/paper/HEARTBEAT_PROMPT.md`
- ✅ Added heartbeat checklist guardrails: `docs/paper/HEARTBEAT_CHECKLIST.md`

---

## 3) TOP GAPS (what still blocks paper quality)

1) **Figures ready for submission:** generate and commit final survival/TOF/recovery plots (SVG/PDF) and ensure captions reference artifacts.
2) **Related work (numbers):** DONE for TRUTH DECAY + Challenging the Evaluator (key effect sizes extracted + cited in Sec 6.4).
3) **Experiment extension decision:** decide whether we need seed5+ (tighter CI) or shift budget to cross-family generalization.

---

## 4) NEXT HEARTBEAT (ONE step)

**Submission-ready figures: generate Survival/TOF/Recovery plots (SVG/PDF) from tracked artifacts and wire them into the draft.**

- Deliverable: add figure files under `docs/paper/figures/` (or `paper_figures/`) + update `docs/paper/PAPER_DRAFT_EN.md` captions to reference them.

---

## 5) Notes / constraints

- Keep runs light (avoid CPU overload; one heavy run at a time).
- If any repo file changes: commit+push (no carry-over ambiguity).
