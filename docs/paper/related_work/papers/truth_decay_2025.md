# TRUTH DECAY: Quantifying Multi-Turn Sycophancy in Language Models

- Slug: truth_decay_2025
- Year: 2025
- Venue: arXiv (also on OpenReview)
- Links:
  - paper: https://arxiv.org/abs/2503.11656
  - openreview: https://openreview.net/forum?id=GHUh9O5Im8
- Bibtex: TODO (grab from arXiv)

## 1) What problem does it study?
Multi-turn 대화에서 LLM이 user feedback/pressure에 의해 **factual accuracy를 희생하며 동조(sycophancy)** 하는 현상을 정량화.

## 2) Experimental setup (what is being measured?)
- Setting: extended dialogues with iterative user challenges/persuasion
- Pressure types: abstract에 따르면 4 types of sycophantic biases를 유도
- Multi-turn: Yes (extended)
- Metrics: multi-turn에서 sycophancy evolution을 측정(구체 metric은 본문 확인 필요)

## 3) Key findings (abstract-level)
- single-turn에서만 보던 sycophancy 분석을 **multi-step으로 확장**
- sycophancy reduction strategies를 제안/평가하며, 단발성 대응이 아니라 **대화 전체에서의 효과**를 본다고 주장

## 4) Limitations / threats
- 아직 우리 쪽에서 full PDF 정독/세부 metric/데이터셋을 구조적으로 정리하지 않음 (TODO)

## 5) How it relates to GALILEO
- What we can cite it for:
  - “multi-turn sycophancy를 별도 벤치마크로 다룬다”는 prior art
- Where we differ (our delta):
  - 우리는 **ground-truth task 기반**으로 survival/TOF/recovery를 측정하고,
  - **Neutral Re-asking Control**로 drift를 분리(pressure 자체 효과 vs 자연 drift).
- Direct mapping:
  - Survival ↔ (그들의 장기 대화 정확도 유지 개념과 대응 가능)
  - TOF ↔ (multi-turn에서 최초로 동조/오답으로 전환되는 시점 개념과 대응 가능)
  - Recovery ↔ (그들이 mitigation/strategy 비교를 한다면 recovery 관점으로 연결 가능)
  - Neutral Re-asking Control ↔ (TRUTH DECAY에는 동일한 drift-control baseline이 있는지 확인 필요)

## 6) Quote-able lines
- Abstract 핵심: “benchmark specifically designed to evaluate sycophancy in extended dialogues … iterative user feedback, challenges, and persuasion.”

## 7) Actions
- [ ] PDF 정독 후: 데이터셋/턴 수/metric/실험 세부를 본 노트에 보강
- [ ] Paper integration: `docs/paper/PAPER_DRAFT_EN.md` Related Work에 1–2문장으로 요약+대조 삽입
- [ ] Add to bib
