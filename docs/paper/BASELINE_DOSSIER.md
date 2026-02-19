# Baseline Dossier (SSOT)

> 목적: **이미 정리된 관련연구**를 “우리 논문에 바로 박을 수 있는 형태”로 정리한다.
> - 각 논문당: Setting / Metrics / What-to-cite / What-to-do (실험/방법론) / Reviewer attack & 대응문장
> - 검색은 금지. `docs/paper/LITERATURE_REVIEW_AND_POSITIONING_KO.md`에 있는 라인업만 기반으로 갱신.

---

## 1) SYCON Bench — Measuring Sycophancy of Language Models in Multi-turn Dialogues (EMNLP 2025 Findings)

- **Problem framing**: multi-turn dialogue에서 사용자 주장에 동조(sycophancy)로 stance가 변하는 동역학 측정.
- **Setting**: 자유형 대화/논쟁 시나리오 중심(정답 GT가 없는/약한 설정이 많음).
- **Core metrics**:
  - *Turn of Flip* (처음 동조로 전환되는 턴)
  - *Number of Flip* (대화 중 stance 변환 횟수)
- **What we cite it for (1–2 sentences)**:
  - “Prior work has proposed multi-turn sycophancy benchmarks and flip-dynamics metrics (e.g., Turn of Flip / number of flips).”
  - “We adapt the flip-dynamics view to **ground-truth tasks**, enabling **auditable scoring** and introducing **survival curves** and **recovery** as complementary dynamics.”
- **Our mapping (explicit aliasing)**:
  - SYCON Turn of Flip ≈ **our TOF (turn-of-failure)**, but ours is defined on *ground-truth correctness*.
  - SYCON #Flips ↔ (optional) our “flip count after failure” (only if we decide to report; otherwise mention qualitatively).
- **Actionable for our paper**:
  - Related Work에 **TOF ↔ Turn of Flip** 한 줄 매핑을 반드시 넣어 reviewer 이해비용 제거.
  - “정답 기반 + survival/recovery”가 왜 추가 기여인지 1문장으로 고정.
- **Reviewer likely attack**: “그럼 SYCON이랑 뭐가 달라?”
  - **Response sentence (drop-in)**: “Unlike SYCON-style open-ended stance tracking, our protocol evaluates multi-turn susceptibility on tasks with verifiable ground truth, allowing cumulative survival analysis and explicit recovery measurement under controlled re-asking baselines.”

---

## 2) TRUTH DECAY — Quantifying Multi-Turn Sycophancy in Language Models (OpenReview)

- **Problem framing**: multi-turn 상호작용에서 누적되는 동조/진실 붕괴를 계량.
- **Setting**: multi-turn pressure/rebuttal류(구체 세팅은 본문 인용 목적에 맞춰 요약만).
- **What we cite it for**:
  - “Multi-turn interaction can progressively erode truthful/correct behavior; benchmarks quantify such decay across turns.”
- **Actionable for our method**:
  - 우리의 **Neutral Re-asking Control**을 ‘단순 multi-turn drift’와 ‘pressure mechanism’을 분리하는 설계로 강조.
- **Reviewer likely attack**: “그냥 multi-turn drift 아닌가?”
  - **Response sentence**: “We explicitly disentangle generic multi-turn drift from pressure-induced failures via a Neutral Re-asking Control, holding turn count and history length constant while removing adversarial framing.”

---

## 3) SycEval — Evaluating LLM Sycophancy (arXiv:2502.08177)

- **Problem framing**: rebuttal을 통해 모델이 사용자의 주장에 동조하며 답을 바꾸는 현상 측정.
- **Setting**: 수학/의료 QA 등(정답 GT 존재)에서 동조/전환 관측.
- **What we cite it for**:
  - “Sycophancy can be elicited by user rebuttals even in ground-truth domains; evaluation protocols measure flips under rebuttal.”
- **Our delta**:
  - 우리는 단발 flip 측정이 아니라 **라운드 기반 survival/TOF/recovery**를 한 프로토콜에 묶고, control로 drift 분리.
- **Actionable**:
  - Related Work에서 “GT 기반 rebuttal → flip 관측”의 선행을 인정하고, “우리는 **dynamics + recovery + control**”로 확장했다고 명시.

---

## 4) Challenging the Evaluator — LLM Sycophancy Under User Rebuttal (arXiv:2509.16533)

- **Key idea**: 동일 내용도 ‘후속 턴 rebuttal 프레이밍’일 때 sycophancy가 커질 수 있다는 framing 효과.
- **What we cite it for**:
  - “User-followup rebuttal framing is a key driver of sycophancy; evaluation design matters.”
- **Actionable**:
  - persona 조건을 “adversarial rebuttal framing”으로, control을 “benign re-asking”으로 정의하여 framing 대비를 명료화.

---

## 5) ELEPHANT — social sycophancy / face-preservation (arXiv:2505.13995)

- **Key idea**: 정답이 없는 open-ended 환경에서의 사회적 동조(체면/관계 유지).
- **What we cite it for**:
  - “Sycophancy is not purely factual error; it can manifest as socially motivated deference/face-preserving responses.”
- **Actionable**:
  - 우리의 qualitative taxonomy(hedging/deference/format compliance 등)를 discussion에서 정당화하는 이론적 배경으로 사용.

---

## 6) Belief vulnerability / Strategic Persuasive Conversation Interventions (SMCR) (arXiv:2601.13590)

- **Key idea**: 설득 전략을 체계화하고, belief 취약성을 대화 전략 관점에서 분석.
- **What we cite it for**:
  - “Persuasion strategies provide a principled axis for categorizing pressure prompts.”
- **Actionable**:
  - 우리의 persona taxonomy를 “message/source 전략” 축으로 재해석해 Related Work/Discussion을 학술적으로 강화.

---

## 7) PERSIST / PTCBench — personality/contextual stability (arXiv:2508.04826, 2602.00016)

- **Key idea**: context/history 변화가 측정된 특성/행동의 instability를 유발.
- **What we cite it for**:
  - “Instability can increase with prompt/history variation; multi-turn settings can amplify such effects.”
- **Actionable**:
  - Intro에서 multi-turn history가 취약성을 키운다는 메타 동기를 제공하고, 우리는 이를 GT 기반 dynamics로 계량한다고 연결.

---

## Appendix: Drop-in comparison table skeleton (fill as we finalize)

| Work | Ground-truth tasks | Multi-turn | Control for drift | Main metric | Recovery | Our relationship |
|---|---:|---:|---:|---|---:|---|
| SYCON | ✗/partial | ✓ | ✗ | Turn of Flip / #Flips | ✗ | TOF aliasing + GT dynamics |
| TRUTH DECAY | mixed | ✓ | partial | decay / sycophancy | ✗ | motivate control + dynamics |
| SycEval | ✓ | mixed | ✗ | flips under rebuttal | ✗ | prior GT flip; we add survival/recovery |
| Challenging Eval | mixed | ✓ | partial | framing effect | ✗ | justify framing + control |
| ELEPHANT | ✗ | mixed | N/A | social sycophancy | N/A | explain social failure modes |
| SMCR persuasion | N/A | ✓ | N/A | persuasion taxonomy | N/A | persona taxonomy rationale |
