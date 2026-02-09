# TRUTH DECAY: Quantifying Multi-Turn Sycophancy in Language Models

- Slug: truth_decay_2025
- Year: 2025
- Venue: arXiv (also on OpenReview)
- Links:
  - paper: https://arxiv.org/abs/2503.11656
  - openreview: https://openreview.net/forum?id=GHUh9O5Im8
- Bibtex:
```bibtex
@misc{liu2025truthdecayquantifyingmultiturn,
      title={TRUTH DECAY: Quantifying Multi-Turn Sycophancy in Language Models}, 
      author={Joshua Liu and Aarav Jain and Soham Takuri and Srihan Vege and Aslihan Akalin and Kevin Zhu and Sean O'Brien and Vasu Sharma},
      year={2025},
      eprint={2503.11656},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.11656}, 
}
```

## 1) What problem does it study?
Multi-turn 대화에서 LLM이 user feedback/pressure에 의해 **factual accuracy를 희생하며 동조(sycophancy)** 하는 현상을 정량화.

## 2) Experimental setup (what is being measured?)
- Core task: **multiple-choice** question answering (variable difficulty) 후, follow-up을 n rounds로 반복.
- Follow-up generation (2 methods):
  - **Static feedback**: 미리 정의한 follow-up template로 “사람이 질문하는 듯한” 반박/피드백을 n회 제공.
  - **Rationale-based feedback**: 별도 모델이 **특정 오답을 지지하는 그럴듯한(하지만 틀린) rationale**을 생성하고, answering model에 반복 제시.
- Sycophantic bias types (static follow-ups; Anthropic single-step test 기반 4종 확장):
  1) Feedback sycophancy (사용자 피드백으로 오답을 밀기)
  2) “Are you sure?” sycophancy (정답을 흔드는 challenge)
  3) Answer sycophancy (다수/외부 출처 의견에 휩쓸리게)
  4) Mimicry sycophancy (사용자가 확신에 찬 ‘사실’로 주장)
- Reduction prompts: follow-up 앞에 붙이는 2개 프롬프트(“Source info”, “Direct command”)를 ablation.
- What is measured: round별 **accuracy 및 response change**를 추적하여 multi-turn에서의 “truth decay/동조 progression”을 계량.

## 3) Models / Datasets (from the paper)
- Models: Claude Haiku, GPT-4o-mini, Llama 3.1 8B Instruct
- Datasets: TruthfulQA, MMLU-Pro

## 4) Key findings (paper-level; with numbers)
- Multi-turn에서 accuracy degradation이 누적됨을 정량적으로 보고. 예: static multistep에서
  - Claude feedback sycophancy accuracy가 **76.74% → 30.23%** (follow-up 7)
  - OpenAI(MMLU-Pro) accuracy가 **49.30% → 26.76%**
  - Llama accuracy가 **29.33% → 5.11%**
  라고 서술 (Sec 5.4).
- (Mitigation) Source Info / Direct Command 같은 reduction prompt를 실험하지만, multi-step에서는 효과가 약해질 수 있다는 framing.
- Rationale-based follow-ups는 단순 동조를 넘어서 “틀린 reasoning internalization”을 유발하며 response instability를 키운다고 주장 (Sec 5.5).

## 5) Limitations / threats
- 우리는 아직 결과 섹션(정량 수치/그래프)을 정독해 “어떤 조건에서 얼마나 악화/완화”를 구체 수치로 인용하지 못함 (TODO: results pass).

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
