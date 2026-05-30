# GALILEO: 대화형 압박 하에서 LLM 정답 유지성 측정

저자: 익명  
문서 상태: 그림 없는 2쪽형 한글 초안  
작성일: 2026-05-30  
근거 범위: 현재 레포의 tracked artifact 기준

## 초록

대규모 언어모델(LLM)은 단일 질문에는 정답을 내더라도, 이후 사용자의 반복 부정·권위 주장·압박성 재질문 속에서 그 정답을 포기할 수 있다. 단일 턴 정확도는 이러한 대화형 붕괴가 언제 시작되는지, 압박이 누적될수록 정답 유지율이 어떻게 변하는지, 한 번 오답으로 전향한 뒤 회복 가능한지를 보여주지 못한다. 본 논문은 ground-truth 과제에서 초기 정답 사례만을 조건부로 고정한 뒤, 5라운드의 persona pressure와 Neutral Re-asking Control(NRC)을 비교하는 GALILEO 프로토콜을 정리한다. GALILEO는 Survival@r, Fail@1/TTF, Recovery@flip을 분리해 보고한다. 현재 tracked artifact 기준으로, seed1–4 aggregated main effect에서 persona pressure는 NRC 대비 Survival@5를 22.76 percentage point 낮추고 Fail@1을 6.93 point 높인다. 또한 Qwen2.5-7B-Instruct seed1–3 authority-pressure 사례에서는 Survival@5 감소가 GSM8K보다 ARC-Easy에서 더 크게 관찰된다. 이 결과는 LLM의 대화형 신뢰성을 단일 정확도가 아니라 정답 유지 궤적과 최초 실패 시점으로 평가해야 함을 시사한다.

**키워드:** LLM evaluation, multi-turn robustness, sycophancy, survival, answer flip, recovery

## 1. 서론

LLM 평가는 흔히 단일 턴 정확도에 의존한다. 그러나 실제 사용 환경에서 대화는 한 번의 질의응답으로 끝나지 않는다. 사용자는 모델의 답을 반복해서 의심하거나, “전문가라면 다른 답을 고른다”와 같은 권위 주장을 제시하거나, 모델이 이전 답을 철회하도록 압박할 수 있다. 이때 중요한 질문은 모델이 처음부터 몰랐는지가 아니라, **처음에는 맞힌 답을 대화 중에도 유지할 수 있는가**이다.

이 문제는 기존 accuracy@1만으로는 포착하기 어렵다. 두 모델의 초기 정확도가 같더라도, 하나는 첫 번째 반박에서 바로 오답으로 전향하고 다른 하나는 여러 턴 동안 정답을 유지할 수 있다. 또한 한 번 오답으로 전향한 뒤 별도의 correction 또는 recovery prompt에 반응해 정답으로 돌아오는지는 또 다른 신뢰성 축이다. 따라서 대화형 환경의 robustness는 한 시점의 정확도가 아니라 시간에 따른 정답 유지 궤적으로 측정해야 한다.

본 논문은 이 목적을 위해 GALILEO 프로토콜을 제안한다. GALILEO는 ground-truth가 명확한 과제에서 초기 정답 사례만을 대상으로 persona pressure와 NRC를 같은 조건에서 비교한다. 이를 통해 단순 반복 질의로 인한 drift와 압박성 발화로 인한 answer flip을 분리하고, multi-turn robustness를 Survival, Fail@1/TTF, Recovery로 나누어 측정한다.

## 2. 방법: GALILEO 프로토콜

GALILEO는 세 단계로 구성된다.

### 2.1 Phase 1: 초기 정답 필터링

모델은 먼저 persona-free 초기 답변을 생성한다. 이후 task-specific evaluator가 정답 여부를 판정한다. GALILEO의 robustness 지표는 기본적으로 초기 정답 사례에 조건부로 계산한다. 이 설계는 base accuracy와 robustness를 분리한다. 즉, 본 프로토콜은 “모델이 원래 몰랐던 문제”가 아니라 “모델이 처음에는 맞혔지만 대화 중 잃어버린 문제”를 측정한다.

### 2.2 Phase 2: Persona pressure와 Neutral Re-asking Control

초기 정답 사례에 대해 R=5 라운드의 follow-up interaction을 수행한다. persona pressure 조건에서는 권위 주장, 강한 압박, 단순 부정, 논리 함정 등 답변 변경을 유도할 수 있는 발화를 사용한다. NRC 조건은 같은 라운드 수와 유사한 응답 조건을 유지하되, 새로운 과제 관련 증거나 대안 답을 주지 않는 중립적 재확인 요청만 사용한다.

핵심 비교는 persona arm과 그에 대응하는 NRC를 **같은 initially-correct subset**에서 평가하는 persona-matched comparison이다. 따라서 Survival@5나 Fail@1의 차이는 서로 다른 분모에서 생긴 selection artifact가 아니라, 동일한 초기 정답 사례 위에서 pressure 조건과 control 조건을 비교한 값으로 해석한다.

### 2.3 Phase 3: Recovery after flip

Phase 2 중 한 번이라도 정답에서 오답으로 전향한 사례를 flip으로 정의한다. 이후 baseline, grounded correction, evidence-bearing 등 설정별 recovery 또는 correction prompt를 적용해 정답으로 돌아오는지를 측정할 수 있다. 다만 본 2쪽 초안의 정량 headline은 Survival@5와 Fail@1에 집중하며, recovery 관련 결과는 “정답 유지성과 회복 가능성은 별도 축으로 측정해야 한다”는 방법론적 역할로만 사용한다.

### 2.4 지표

| 지표 | 정의 | 해석 |
|---|---|---|
| Survival@r | 초기 정답 사례 중 round r까지 모든 턴에서 정답을 유지한 비율 | 압박 누적에 대한 정답 유지성 |
| Fail@1 / TTF | 첫 번째 압박 턴에서 실패한 비율 / 최초 실패 시점 | 초기 압박 취약성 |
| Recovery@flip | Phase 2에서 flip된 사례 중 recovery/correction prompt 후 정답으로 돌아온 비율 | 오답 전향 이후 회복 가능성 |

Survival@r은 round r에서의 단순 정확도가 아니라, round 1부터 r까지 한 번도 실패하지 않은 누적 생존율이다. 따라서 중간에 오답으로 전향했다가 이후 다시 정답을 낸 사례는 Survival에는 실패로 기록되고, 회복 여부는 Recovery@flip에서 별도로 측정된다.

## 3. 실험 설정 및 보고 범위

본 초안은 현재 레포에 tracked artifact로 남아 있는 결과만 사용한다. 주요 결과는 두 범위로 나누어 보고한다.

첫째, aggregated main effect는 `table_w_effect_delta_seed1-4_20260209.csv`에 저장된 seed1–4 결과를 사용한다. 이 표는 persona-matched comparison을 기반으로 하며, persona side는 persona-weighted pooled aggregate로 요약되어 있다. 따라서 본문 표의 control mean, persona mean, delta는 해당 artifact의 집계 규칙을 따른다.

둘째, Qwen2.5-7B-Instruct case study는 GSM8K와 ARC-Easy에서 authority pressure와 NRC의 Survival@5 차이를 seed1–3 평균으로 보고한다. 여기서 evidence baseline과 grounded baseline은 해당 Qwen7B multiseed artifact 패키지의 두 correction/recovery 설정을 가리키며, 본문에서는 이들 설정에서 관찰된 authority-vs-control Survival@5 차이만 사용한다.

이 범위 때문에 본 논문의 실증 주장은 “모든 모델과 모든 과제에서의 일반 법칙”이 아니라, 현재 tracked artifact가 뒷받침하는 보수적 관찰로 제한한다.

## 4. 결과

### 4.1 Persona pressure는 Survival@5를 낮추고 Fail@1을 높인다

Aggregated main effect에서 persona pressure는 NRC 대비 Survival@5를 낮추고 Fail@1을 높인다. 아래 표는 `table_w_effect_delta_seed1-4_20260209.csv`의 persona-weighted pooled aggregate 기준이다.

| Metric | NRC mean | Persona mean | Persona − NRC |
|---|---:|---:|---:|
| Survival@5 | 80.32% | 57.55% | **-22.76pp** |
| Fail@1 | 13.10% | 20.03% | **+6.93pp** |
| Never-fail | 80.32% | 57.86% | **-22.46pp** |

이 결과는 R=5 horizon에서 persona pressure 조건의 누적 정답 유지율이 NRC보다 낮고, 첫 번째 압박 턴에서의 실패율은 더 높다는 것을 보여준다. 즉, 동일한 initially-correct subset 위에서 중립적 재확인만 반복한 조건보다 압박성 follow-up 조건에서 answer flip이 더 자주 관찰된다.

근거 artifact: `docs/paper/artifacts/table_w_effect_delta_seed1-4_20260209.csv`

### 4.2 Qwen2.5-7B-Instruct 사례에서 ARC-Easy 감소 폭이 더 크게 관찰된다

Qwen2.5-7B-Instruct seed1–3 artifact에서는 authority pressure의 Survival@5 감소가 GSM8K보다 ARC-Easy에서 더 크게 관찰된다.

| Setting | Dataset | ΔSurvival@5 | 95% CI |
|---|---|---:|---:|
| Evidence baseline | GSM8K | **-0.215** | [-0.293, -0.167] |
| Evidence baseline | ARC-Easy | **-0.674** | [-0.755, -0.633] |
| Grounded baseline | GSM8K | **-0.208** | [-0.341, -0.070] |
| Grounded baseline | ARC-Easy | **-0.612** | [-0.653, -0.551] |

이 표는 현재 Qwen2.5-7B-Instruct multiseed package 안에서 ARC-Easy가 GSM8K보다 authority pressure에 따른 Survival@5 감소가 크게 나타났음을 보여준다. 다만 이는 단일 모델 family와 두 데이터셋에 대한 관찰이므로, 객관식 또는 비수학형 과제가 일반적으로 더 취약하다고 단정하지 않는다. 더 넓은 일반화를 위해서는 추가 모델 family와 과제군에서 같은 persona-matched protocol을 반복해야 한다.

근거 artifact:

- `docs/paper/artifacts/qwen7b_evidence_multiseed_seed1-3_deltas_20260310.csv`
- `docs/paper/artifacts/qwen7b_grounded_multiseed_seed1-3_deltas_20260310.csv`

## 5. 논의

본 결과의 핵심 함의는 LLM의 대화형 신뢰성을 단일 턴 정확도로만 평가해서는 안 된다는 점이다. 초기 정답률이 높은 모델이라도, 사용자의 반복 부정이나 권위 주장에 의해 정답에서 이탈할 수 있다. 따라서 실제 배포 환경에서는 모델이 정답을 알고 있는지뿐 아니라, 대화 압박 속에서도 그 정답을 유지할 수 있는지를 측정해야 한다.

GALILEO의 NRC는 이 해석에서 중요하다. multi-turn interaction 자체는 일반적인 drift를 만들 수 있으므로, persona pressure의 효과를 주장하려면 같은 라운드 수와 유사한 응답 조건을 가진 control이 필요하다. GALILEO는 persona arm과 NRC를 같은 initially-correct subset에서 비교함으로써 반복 대화의 일반 효과와 압박성 발화의 효과를 분리한다.

또한 Survival, Fail@1/TTF, Recovery@flip은 서로 다른 현상을 포착한다. Survival은 누적 정답 유지성을, Fail@1은 첫 반박에 대한 취약성을, Recovery는 오답 전향 이후 복구 가능성을 측정한다. 본 초안의 headline 결과는 Survival과 Fail@1에 집중하지만, recovery 축을 별도로 정의하는 것은 “끝까지 버티는 능력”과 “무너진 뒤 돌아오는 능력”을 혼동하지 않기 위해 필요하다.

## 6. 한계

첫째, 본 초안의 가장 강한 정량 주장은 현재 tracked artifact, 특히 Qwen2.5-7B-Instruct와 일부 aggregated artifact에 기반한다. 따라서 모든 모델 family에 대한 일반 명제로 과장해서는 안 된다.

둘째, 본문 Qwen case study는 GSM8K와 ARC-Easy 두 데이터셋에 집중한다. 과제 형식이나 도메인에 따른 취약성 차이는 관찰되지만, 이를 일반화하려면 더 많은 task family와 model family가 필요하다.

셋째, 모든 robustness 지표는 초기 정답 사례에 조건부로 계산된다. 이는 “처음에는 맞힌 답을 유지하는가”라는 질문에는 적합하지만, base capability와 직접 비교되는 전체 accuracy 지표는 아니다. 따라서 model-family 간 비교에서는 초기 정확도와 조건부 robustness를 함께 보고해야 한다.

넷째, 본 프로토콜은 ground-truth가 있는 과제에 초점을 둔다. 주관적 판단, 장문 생성, 가치 판단 과제에서는 정답 판정 자체가 달라지므로 별도의 평가 설계가 필요하다.

## 7. 결론

GALILEO는 LLM의 multi-turn correctness를 단일 정확도가 아니라 시간에 따른 정답 유지, 최초 실패 시점, 회복 가능성으로 측정하는 평가 프로토콜이다. 현재 tracked artifact는 persona pressure가 NRC 대비 Survival@5를 낮추고 Fail@1을 높인다는 근거를 제공한다. 또한 Qwen2.5-7B-Instruct 사례에서 GSM8K와 ARC-Easy의 감소 폭 차이는 대화형 robustness가 과제별로 다르게 나타날 수 있음을 시사한다. 따라서 LLM의 실제 대화형 신뢰성을 평가하려면 초기 정답률뿐 아니라, 압박 속에서의 survival trajectory와 failure timing을 함께 보고해야 한다.

## 재현 근거

본문 수치는 다음 tracked artifact에서 가져왔다.

- `docs/paper/artifacts/table_w_effect_delta_seed1-4_20260209.csv`
- `docs/paper/artifacts/qwen7b_evidence_multiseed_seed1-3_deltas_20260310.csv`
- `docs/paper/artifacts/qwen7b_grounded_multiseed_seed1-3_deltas_20260310.csv`

프로토콜과 지표 정의는 다음 문서를 따른다.

- `docs/paper/FIGURE_CAPTIONS.md`
- `docs/paper/CLAIM_EVIDENCE_MAP.md`
- `docs/paper/PAPER_EVIDENCE_STATUS_20260310.md`
