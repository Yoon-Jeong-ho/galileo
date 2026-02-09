# Time-To-Inconsistency: A Survival Analysis of Large Language Model Robustness to Adversarial Attacks

- Slug: time_to_inconsistency_2025
- Year: 2025 (latest arXiv versions go into 2026, but initial appears 2025)
- Venue: arXiv
- Links:
  - paper: https://arxiv.org/abs/2510.02712
- Bibtex: TODO (grab from arXiv)

## 1) What problem does it study?
Multi-turn dialogue robustness를 “time-to-event”로 보고 **survival analysis**로 inconsistency failure를 모델링.

## 2) Experimental setup (from abstract)
- Data: 36,951 turns, 9 LLMs, MT-Consistency benchmark
- Multi-turn: Yes
- Attacks: adversarial / semantic drift patterns
- Methods: Cox PH, AFT, Random Survival Forest; drift features로 hazard를 설명
- Output: turn-level risk monitor (몇 턴 전에 failure flag)

## 3) Key findings (abstract-level)
- abrupt semantic drift가 hazard를 크게 증가
- cumulative drift는 오히려 보호적일 수 있다는 관찰(적응 가설)
- AFT + model-drift interaction이 discrimination+calibration에서 좋음

## 4) Limitations / threats
- task가 “inconsistency” 중심이라, 우리처럼 ground-truth correctness/attack family/recovery로 쪼개는 것과는 결이 다를 수 있음.

## 5) How it relates to GALILEO
- What we can cite it for:
  - multi-turn robustness를 survival/time-to-failure로 정량화한 prior
- Where we differ (our delta):
  - 우리는 **ground-truth tasks**에서 correctness 기반의 survival/TOF를 정의하고,
  - **Recovery + Neutral control**을 포함하여 ‘pressure effect vs drift’를 분해.
- Direct mapping:
  - Survival ↔ time-to-inconsistency
  - TOF ↔ first inconsistency turn
  - Recovery ↔ (그들의 risk monitor와 다르게 우리는 flip 이후 correctness 복귀를 직접 측정)
  - Neutral control ↔ (drift-only baseline이 있는지 확인 필요)

## 6) Actions
- [ ] 우리 Related Work에 survival-analysis prior로 1문장 삽입(“time-to-event framing exists; we specialize to correctness+persona pressure+recovery+neutral control”)
