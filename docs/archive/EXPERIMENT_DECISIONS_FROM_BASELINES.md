# Experiment Decisions from Baselines (SSOT)

> 목적: 관련연구를 ‘인용’으로 끝내지 않고, **실험/방법론/서술 의사결정**으로 고정한다.
> 원칙: 한 항목은 (i) baseline-motivated risk, (ii) decision, (iii) evidence/artifact pointer로 구성.

---

## D1. SYCON Bench 대응: TOF/Flip naming 충돌 방지 + 명시적 매핑
- **Risk (reviewer)**: “TOF가 기존 Turn of Flip이랑 다른 말장난 아닌가?”
- **Decision**:
  - Related Work에서 **Turn of Flip ↔ TOF**를 ‘closest analogue’로 1문장 매핑.
  - 단, 우리는 **ground-truth correctness 기반**이라 score가 audit 가능하다고 차별점 명시.
- **Where to implement**:
  - `docs/paper/PAPER_DRAFT_EN.md`: Related Work + Metrics definition
  - `docs/paper/FIGURE_CAPTIONS.md`: TOF 정의에 alias 언급(각주 수준 가능)
  - `docs/paper/CLAIM_EVIDENCE_MAP.md`: novelty bullet에 반영

---

## D2. TRUTH DECAY / Challenging the Evaluator 대응: drift vs pressure 분리(Neutral Re-asking Control을 ‘기여’로 격상)
- **Risk**: “그냥 multi-turn 대화가 길어지면 틀리는 drift 아닌가?”
- **Decision**:
  - Control을 단순 baseline이 아니라 **design contribution**으로 서술.
  - 모든 주요 결과(그림/표)에서 control 대비를 최소 1회 이상 명시.
- **Evidence pointers (existing)**:
  - `docs/paper/artifacts/`의 control vs persona CSV들
  - `docs/paper/figures/`의 control vs persona survival/Δ 지표

---

## D3. SycEval 대응: ‘GT 도메인에서 rebuttal flip’은 선행 인정 + 우리는 dynamics/survival/recovery로 확장
- **Risk**: “GT QA에서 rebuttal로 flip 보는 건 이미 했다(SycEval).”
- **Decision**:
  - SycEval을 정면으로 인용하고, “우리는 (i) 라운드 기반 survival curve, (ii) TOF 분포, (iii) recovery를 추가하고, (iv) control로 drift를 분리”라고 한 문장으로 정리.

---

## D4. ELEPHANT/SMCR persuasion 대응: qualitative taxonomy를 ‘임의적’이 아니라 ‘이론/전략 축’으로 정당화
- **Risk**: “persona가 그냥 prompt 변형이고 taxonomy가 임의적이다.”
- **Decision**:
  - Discussion에서 ELEPHANT(사회적 동조) + SMCR(설득 전략)을 ‘분류 축’으로 언급.
  - taxonomy 섹션을 “face-preservation / authority pressure / denial framing …”처럼 baseline 용어와 1:1로 연결.

---

## D5. Cross-family generalization: baseline들이 요구하는 최소 일반성 기준 충족
- **Risk**: “Qwen-specific 현상으로 보인다.”
- **Decision**:
  - Tier-1 cross-family(이미 확보된 Llama/Phi/Mistral/Qwen 등) 결과는 Related Work의 ‘일반성’ 기대에 대한 직접 대응으로 배치.
- **Evidence pointers**:
  - `docs/paper/figures/` cross-family survival figure
  - `docs/paper/artifacts/` tier1_*_survival_summary CSV

