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
- ✅ Auditable pipeline smoke: on nlp8, `results/smoke_20260209_162417/` produced:
  - `paper_exports/{survival_curve,turn_of_failure,flip_samples}.csv`
  - `paper_exports/{metadata.json,runner_metadata.json}`
  - validator printed `[OK]`.

---

## 2) What is NOT DONE (top gaps)

1) **Control vs persona 2-run (fresh) (paper-ready)**: control is DONE ✅, persona still running ⏳ → once persona finishes, refresh Table W.
2) **GLOBAL_VALIDATE.log** generation integrated for new runs (runner-level global validate).
3) **Paper integration**: SYCON/TRUTH DECAY/rebuttal framing 내용을 `PAPER_DRAFT_EN.md` 본문 Related Work에 완전히 이식.

---

## 3) Next heartbeat (ONE step)

**Finish fresh persona run + validate exports (paper-ready), then prep Table W refresh.**

- Must for persona run:
  - produce `paper_exports/*` + `metadata.json` + `runner_metadata.json`
  - validator prints `[OK]` (exports + runner_metadata parity)
- Then (same heartbeat if quick, otherwise next): update Table W inputs to point at the new control+persona run roots.

---

## 4) Notes / constraints

- Keep runs light (avoid CPU overload, one heavy run at a time).
- Every experiment must have a short rationale committed to git (why we ran it, what decision it informs).
