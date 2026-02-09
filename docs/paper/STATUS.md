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

## 1) What is DONE (verifiable)

- ✅ Paper terminology: drift baseline is standardized as **Neutral Re-asking Control** in the draft.
- ✅ Export standardization: paper export normalizes control labels to `neutral_reask_control`.
- ✅ Auditable pipeline smoke (nlp8): `results/smoke_20260209_162417/` produced `paper_exports/*` + `metadata.json` + `runner_metadata.json` and validator printed `[OK]`.
- ✅ Fresh control vs persona seed1 pair is auditable green (nlp8):
  - Control: `results/c2run_control_20260209_172640/`
  - Persona: `results/c2run_persona_20260209_174640/`
- ✅ Fresh control vs persona seed2 pair is auditable green (nlp8):
  - Control: `results/c2run_control_seed2_20260209_194621/`
  - Persona: `results/c2run_persona_seed2_20260209_200611/`
- ✅ **Table W seed1–2 artifacts + aggregate landed in the draft:**
  - Artifacts: `docs/paper/artifacts/table_w_control_vs_persona_seed{1,2}_20260209.csv`
  - Aggregate: `docs/paper/artifacts/table_w_control_vs_persona_seed1-2_mean_std_20260209.csv`
  - Draft block: `docs/paper/PAPER_DRAFT_EN.md` (AUTO:TABLE_W_SEED12)

---

## 2) What is NOT DONE (top gaps)

1) **Table W multi-seed**: extend from seed1–2 to seed1–3+ (std 안정화) and keep the draft AUTO block updated.
2) **GLOBAL_VALIDATE.log** generation integrated for new runs (runner-level global validate).
3) **Paper integration**: SYCON/TRUTH DECAY/rebuttal framing 내용을 `PAPER_DRAFT_EN.md` Related Work에 “정식 문장+인용”으로 완전히 이식.

---

## 3) Next heartbeat (ONE step)

**Run seed3 control+persona (auditable green) to stabilize Table W (seed1–3 aggregate).**

- Target:
  - control seed3 on an idle GPU (0–3)
  - persona seed3 on another idle GPU (0–3)
- Must:
  - each run produces `paper_exports/*` + `metadata.json` + `runner_metadata.json`
  - validator prints `[OK]` and `[OK] runner_metadata parity

---

## 4) Notes / constraints

- Keep runs light (avoid CPU overload, one heavy run at a time).
- Every experiment must have a short rationale committed to git (why we ran it, what decision it informs).
