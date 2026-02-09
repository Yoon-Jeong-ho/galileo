# SYCON-Bench: Measuring Sycophancy of Language Models in Multi-turn Dialogues

- Slug: sycon_bench_2025
- Year: 2025
- Venue: EMNLP 2025 Findings (per repo badge) / arXiv
- Links:
  - repo: https://github.com/JiseungHong/SYCON-Bench
  - pdf (repo): https://arxiv.org/pdf/2505.23840
  - paperswithcode: https://paperswithcode.com/paper/measuring-sycophancy-of-language-models-in
- Bibtex: TODO (grab from arXiv/ACL)

## 1) What problem does it study?
Multi-turn free-form 대화에서 모델이 user pressure에 **얼마나 빨리 동조/전환(sycophancy)** 하는지 측정하는 벤치마크.

## 2) Experimental setup (from repo overview)
- Settings (3):
  1) Debate (controversial topics + predefined stance)
  2) Ethical (StereoSet 기반 harmful stereotype pressure)
  3) False presuppositions (factually incorrect assumptions under pressure)
- Multi-turn: Yes
- Metrics:
  - Turn of Flip (ToF): 얼마나 빨리 flip(동조/전환)하는지
  - Number of Flips (NoF): sustained pressure 하에서 stance가 얼마나 자주 바뀌는지

## 3) Key findings
- (TODO: 논문 정독 필요) repo에는 주로 benchmark 설계/측정치 소개가 요약되어 있음.

## 4) Limitations / threats
- setting이 stance/윤리/false presupposition 중심. 우리처럼 **ground-truth task 정확도 유지/붕괴/회복**을 직접 계량화하는 구조와는 차이가 있음.

## 5) How it relates to GALILEO
- What we can cite it for:
  - multi-turn sycophancy의 대표 benchmark + ToF라는 용어/측정 관례
- Where we differ (our delta):
  - 우리는 ground-truth tasks에서 correctness 기반 **Survival/TOF/Recovery**를 측정하고,
  - drift를 분리하기 위해 **Neutral Re-asking Control**을 포함.
- Direct mapping:
  - Survival ↔ 그들의 “stance 유지/flip 없음”과 느슨하게 대응
  - TOF ↔ (정확히 용어가 일치: Turn of Flip)
  - Recovery ↔ (우리만의 강한 차별점: flip 이후 정답 복귀 측정)
  - Neutral control ↔ (SYCON은 baseline prompt strategies는 있으나 drift-control로서 neutral re-asking이 있는지 확인 필요)

## 6) Quote-able lines (repo)
- “measures how quickly a model conforms to the user (Turn of Flip) and how frequently it shifts its stance … (Number of Flip)”

## 7) Actions
- [ ] arXiv/ACL에서 bibtex 확보
- [ ] 논문 정독 후: 세부 metric/데이터/턴 수/실험 프로토콜을 이 노트에 보강
- [ ] 우리 Related Work에서 “TOF 용어 매핑(SYCON ToF ↔ GALILEO TOF)”를 명시
