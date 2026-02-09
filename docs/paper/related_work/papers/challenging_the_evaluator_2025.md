# Challenging the Evaluator: LLM Sycophancy Under User Rebuttal

- Slug: challenging_the_evaluator_2025
- Year: 2025
- Venue: EMNLP 2025 Findings (also arXiv)
- Links:
  - paper: https://arxiv.org/abs/2509.16533
  - acl anthology (Findings): https://aclanthology.org/2025.findings-emnlp.1222/
- Bibtex: TODO (grab from ACL Anthology)

## 1) What problem does it study?
대화에서 user rebuttal(후속 반박/피드백) 프레이밍이 LLM의 **sycophancy**를 촉발하는 현상.
특히 “동시에 비교 평가”에서는 잘 판단하는데 “follow-up user rebuttal” 형태에서는 쉽게 설득되는 긴장을 다룸.

## 2) Experimental setup (what is being measured?)
- Interaction patterns: follow-up rebuttal vs simultaneous evaluation framing
- Multi-turn: Yes (subsequent conversational turns)
- Cues tested (abstract): detailed reasoning, casual vs formal critique, follow-up framing
- Metrics: user counterargument endorsement / persuasion susceptibility (정확한 정의는 본문 확인 필요)

## 3) Key findings (from abstract)
- follow-up user로 프레이밍되면 더 쉽게 counterargument를 채택
- rebuttal이 자세한(하지만 틀린 결론의) reasoning을 포함하면 더 취약
- 캐주얼한 피드백이 포멀한 critique보다 더 잘 흔드는 경우가 있음

## 4) Limitations / threats
- 우리 쪽에서 아직 full-text 기반으로 “metric/데이터/턴 구조”를 정리하지 않음 (TODO)

## 5) How it relates to GALILEO
- What we can cite it for:
  - “multi-turn에서 프레이밍/후속 반박이 sycophancy를 강화”한다는 실증 근거
- Where we differ (our delta):
  - 우리는 ground-truth task에서 **정답 유지(생존) / 붕괴 시점(TOF) / 붕괴 후 회복**을 계량화하고,
  - **Neutral Re-asking Control**로 drift baseline을 제공.
- Direct mapping:
  - Survival ↔ rebuttal turns에서 correctness 유지
  - TOF ↔ rebuttal 이후 ‘최초 endorsement/오답’ 타이밍으로 재해석 가능
  - Recovery ↔ (우리의 recovery는 flip 이후 정답 복귀; 이 논문에서는 persuasion 후 재복귀를 다루는지 확인)
  - Neutral control ↔ (이 논문에 동일 baseline이 있는지 확인 필요)

## 6) Quote-able lines
- Abstract: “more likely to endorse a user's counterargument when framed as a follow-up … than when both responses are presented simultaneously for evaluation”

## 7) Actions
- [ ] ACL Anthology에서 bibtex 확보
- [ ] Related Work 문단에 rebuttal-framing prior로 인용 + 우리 framing(TOF/Neutral control)과 차별점 명시
