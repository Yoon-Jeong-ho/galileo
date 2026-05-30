# GALILEO 2쪽 논문 가능성 README

작성일: 2026-05-30  
범위: 현재 로컬 체크아웃(`/data_x/aa007878/projects/galileo`) 기준 읽기 전용 점검 요약

## 결론

**2쪽짜리 짧은 논문, 워크숍 페이퍼, 포스터 요약으로 충분히 작성 가능하다.**

가장 안전한 포지셔닝은 “새로운 해결책을 완성했다”가 아니라 다음이다.

> **GALILEO는 LLM이 처음에는 맞힌 정답을 대화형 압박 속에서 얼마나 오래 유지하고, 언제 실패하며, 한 번 틀린 뒤 회복하는지를 측정하는 drift-controlled multi-turn evaluation protocol이다.**

현재 레포에는 이미 다음 요소가 있다.

- 명확한 연구 질문
- 3단계 평가 프로토콜
- Neutral Re-asking Control, 즉 drift baseline
- Survival@r, Fail@1/TTF, Recovery@flip 지표
- 논문용 CSV artifact와 SVG 그림
- 영어/한국어 논문 초안 및 claim-evidence map

다만 현재 체크아웃에서는 일부 raw result root나 `results_paper/` 링크가 비어 있거나 깨져 있으므로, 2쪽 버전은 **tracked artifact 중심**으로 쓰는 것이 안전하다.

## 추천 제목

1. **GALILEO: Measuring Truth Survival under Multi-turn Persona Pressure**
2. **Beyond Single-turn Accuracy: Drift-controlled Evaluation of Pressure-induced Answer Flips**
3. **When Correct Answers Erode: Measuring Failure Timing and Recovery in LLM Dialogue**

가장 추천하는 제목은 1번이다.

## 논문 동기 초안

### 한국어

LLM의 정확도는 보통 단일 턴에서 측정되지만, 실제 사용에서는 사용자가 반복적으로 부정하거나 권위에 호소하거나 다른 답을 유도할 수 있다. 이때 모델이 처음에는 맞힌 답을 언제 포기하는지, 압박이 누적될수록 정답 유지율이 어떻게 감소하는지, 한 번 틀린 뒤 다시 회복 가능한지는 단일 턴 정확도로는 보이지 않는다. GALILEO는 ground-truth 과제에서 이러한 대화형 정답 붕괴를 Survival, Fail@1, Recovery로 계량한다.

### English

Single-turn accuracy does not reveal whether an LLM can preserve a correct answer under conversational pressure. In real interactions, users may repeatedly deny, challenge, or reframe a correct response without providing new evidence. GALILEO measures this erosion directly by tracking survival, first-turn failure, and recovery under persona pressure relative to a neutral re-asking control.

## 핵심 기여

### 1. 평가 프로토콜

초기 정답 여부를 먼저 고정한 뒤, persona pressure와 neutral re-asking control을 비교하고, flip 이후 recovery까지 측정하는 **3-phase multi-turn evaluation protocol**을 제안한다.

근거 파일:

- `run_experiment.py`
- `personas.py`
- `docs/paper/figures/protocol_overview.svg`

### 2. 지표 설계

정답 붕괴를 단일 정확도가 아니라 다음 trajectory 지표로 측정한다.

- **Survival@r**: round r까지 계속 정답을 유지한 비율
- **Fail@1 / TTF**: 첫 압박 턴에서 바로 실패하는 비율 / 최초 실패 시점
- **Recovery@flip**: 한 번 오답으로 flip된 뒤 중립 recovery prompt에서 정답으로 돌아오는 비율

근거 파일:

- `docs/paper/FIGURE_CAPTIONS.md`
- `scripts/paper_export.py`

### 3. 실증 결과

현재 tracked artifact 기준으로 persona pressure는 neutral control 대비 Survival@5를 낮추고 Fail@1을 높인다. 특히 Qwen7B 사례에서는 ARC-Easy에서 GSM8K보다 압박 효과가 더 크게 나타난다.

근거 파일:

- `docs/paper/artifacts/table_w_effect_delta_seed1-4_20260209.csv`
- `docs/paper/artifacts/qwen7b_evidence_multiseed_seed1-3_deltas_20260310.csv`
- `docs/paper/artifacts/qwen7b_grounded_multiseed_seed1-3_deltas_20260310.csv`

### 4. 감사 가능한 논문 파이프라인

실험 로그에서 paper-ready CSV, 그림, 검증 스크립트로 이어지는 auditable export pipeline이 있다.

근거 파일:

- `scripts/validate_paper_exports.py`
- `scripts/make_paper_figures_from_artifacts.py`
- `docs/paper/CLAIM_EVIDENCE_MAP.md`

## 바로 쓸 수 있는 결과

### Headline result 1: aggregated main effect

원천 파일:

- `docs/paper/artifacts/table_w_effect_delta_seed1-4_20260209.csv`

| Metric | Persona − Control |
|---|---:|
| Survival@5 | **-22.76pp** |
| Fail@1 | **+6.93pp** |

논문 문장 예:

> Across the aggregated main setting, persona pressure reduces Survival@5 by 22.76 percentage points and increases first-turn failure by 6.93 points relative to neutral re-asking control.

### Headline result 2: Qwen7B multiseed case study

원천 파일:

- `docs/paper/artifacts/qwen7b_evidence_multiseed_seed1-3_deltas_20260310.csv`
- `docs/paper/artifacts/qwen7b_grounded_multiseed_seed1-3_deltas_20260310.csv`
- `docs/paper/artifacts/qwen7b_threeway_multiseed_comparison_20260310.csv`

| Setting | GSM8K ΔSurvival@5 | ARC-Easy ΔSurvival@5 |
|---|---:|---:|
| Evidence baseline | **-0.215** | **-0.674** |
| Grounded baseline | **-0.208** | **-0.612** |

해석:

- persona pressure는 두 과제 모두에서 Survival@5를 낮춘다.
- ARC-Easy에서 감소 폭이 더 커서, task/domain에 따라 압박 취약성이 크게 달라진다는 메시지를 쓸 수 있다.

## 추천 그림과 표

2쪽이면 **그림 1개 + 표 1개**가 가장 읽기 좋다.

### 그림 1: 프로토콜 개요

- 파일: `docs/paper/figures/protocol_overview.svg`
- 용도: GALILEO 3-phase 평가 구조 설명

포함할 내용:

1. initial correctness filtering
2. persona pressure vs neutral re-asking control
3. recovery after flip

### 표 1: 핵심 결과 요약

권장 원천:

- `docs/paper/artifacts/table_w_effect_delta_seed1-4_20260209.csv`
- `docs/paper/artifacts/qwen7b_evidence_multiseed_seed1-3_deltas_20260310.csv`

2쪽용 표 예:

| Result | Value |
|---|---:|
| Survival@5, persona − control | -22.76pp |
| Fail@1, persona − control | +6.93pp |
| GSM8K ΔSurvival@5, Qwen7B evidence | -0.215 |
| ARC-Easy ΔSurvival@5, Qwen7B evidence | -0.674 |

### 선택 그림: dynamics

공간이 있으면 다음 중 하나를 작은 보조 그림으로 넣을 수 있다.

- `docs/paper/figures/survival_curves_rounds_seed1-4_20260209.svg`
- `docs/paper/figures/tof_personawise_fail1_delta_seed1-4_20260209.svg`
- `docs/paper/figures/table_w_effect_delta_seed1-4_20260209.svg`

## 2쪽 구성안

### Page 1

1. **Title + short abstract**
   - GALILEO: Measuring Truth Survival under Multi-turn Persona Pressure
2. **Motivation**
   - 단일 턴 정확도는 대화 중 정답 붕괴를 설명하지 못한다.
3. **Protocol**
   - 프로토콜 그림 1개
   - 지표 3개: Survival@r, Fail@1/TTF, Recovery@flip

### Page 2

1. **Main results**
   - 핵심 결과 표 1개
   - Qwen7B GSM8K vs ARC-Easy case study
2. **Optional qualitative example**
   - `docs/paper/PAPER_RESULTS_QUAL_EXAMPLES_KO.md`에서 flip 사례 1개
3. **Limitations**
   - strongest evidence는 현재 Qwen7B artifact 중심
   - 일부 raw result root와 `results_paper/` 링크는 현재 체크아웃에서 불완전
   - `evidence_gate`는 해결책이 아니라 trade-off discussion으로만 언급
4. **Conclusion**
   - multi-turn robustness는 single number가 아니라 trajectory로 측정해야 한다.

## 피해야 할 과장

아래 표현은 피하는 것이 좋다.

- “GALILEO가 압박 취약성을 해결했다.”
- “evidence_gate가 일반적 mitigation으로 검증됐다.”
- “모든 모델 family에서 완전히 동일한 효과가 강하게 입증됐다.”
- “현재 체크아웃만으로 모든 raw experiment를 즉시 재생성할 수 있다.”

더 안전한 표현은 다음이다.

- “GALILEO provides a measurement protocol.”
- “Tracked artifacts show a consistent pressure-induced survival drop in the Qwen7B case study.”
- “Evidence-gated prompting is a promising but trade-off-bearing mitigation candidate.”
- “The short paper reports artifact-backed results and explicitly notes raw-result synchronization limitations.”

## 제출 전 최소 보강 체크리스트

2쪽 초안을 제출하거나 공유하기 전에 아래를 확인하면 좋다.

- [ ] `docs/paper/artifacts/table_w_effect_delta_seed1-4_20260209.csv` 수치 재확인
- [ ] `docs/paper/artifacts/qwen7b_*_deltas_20260310.csv` 수치 재확인
- [ ] `docs/paper/figures/protocol_overview.svg`가 최신 프로토콜과 일치하는지 확인
- [ ] `results_paper/` raw roots 또는 symlink 상태 복구 여부 확인
- [ ] 익명 제출이면 절대경로/호스트명 제거
- [ ] `evidence_gate`를 headline claim이 아니라 limitation/discussion으로 이동

## 근거 파일 요약

핵심 코드:

- `run_experiment.py`
- `personas.py`
- `data_loader.py`
- `tasks.py`
- `evaluation.py`
- `inference.py`

핵심 문서:

- `README.md`
- `docs/paper/PAPER_DRAFT_EN.md`
- `docs/paper/PAPER_DRAFT_KO.md`
- `docs/paper/FIGURE_CAPTIONS.md`
- `docs/paper/CLAIM_EVIDENCE_MAP.md`
- `docs/paper/PAPER_EVIDENCE_STATUS_20260310.md`

핵심 artifacts:

- `docs/paper/artifacts/table_w_effect_delta_seed1-4_20260209.csv`
- `docs/paper/artifacts/qwen7b_evidence_multiseed_seed1-3_deltas_20260310.csv`
- `docs/paper/artifacts/qwen7b_grounded_multiseed_seed1-3_deltas_20260310.csv`
- `docs/paper/artifacts/qwen7b_threeway_multiseed_comparison_20260310.csv`

핵심 figures:

- `docs/paper/figures/protocol_overview.svg`
- `docs/paper/figures/survival_curves_rounds_seed1-4_20260209.svg`
- `docs/paper/figures/table_w_effect_delta_seed1-4_20260209.svg`

## 최종 한 문장

**이 레포는 2쪽짜리 짧은 논문으로 만들 수 있으며, 가장 강한 이야기는 “LLM의 multi-turn correctness는 단일 정확도가 아니라 survival, failure timing, recovery로 측정해야 한다”는 평가 프로토콜 논문이다.**
