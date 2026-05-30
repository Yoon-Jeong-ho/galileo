# GALILEO: 대화형 압박에서 LLM의 정답 유지와 초기 실패 취약성 측정

저자: 익명  
문서 상태: 그림 없는 2쪽형 한글 초안  
작성일: 2026-05-30  
근거 범위: 현재 레포의 tracked artifact 기준

## 초록

대규모 언어모델(LLM)은 처음에는 맞힌 답도 사용자의 반복 부정이나 권위 주장 속에서 바꿀 수 있지만, 단일 턴 정확도는 이러한 대화형 붕괴가 언제 시작되고 얼마나 누적되는지 보여주지 못한다. 본 논문은 초기 정답 사례를 기준으로 5라운드 persona pressure와 Neutral Re-asking Control(NRC)을 비교하는 GALILEO 프로토콜을 정리하고, Survival@5와 Fail@1로 정답 유지성과 초기 실패 취약성을 측정한다. Qwen2.5-7B-Instruct seed1–4 snapshot에서 persona pressure는 NRC 대비 Survival@5를 22.76%p 낮추고 Fail@1을 6.93%p 높였다. 이 결과는 LLM의 대화형 신뢰성을 단일 정확도보다 정답 유지 궤적과 초기 실패 취약성으로 함께 평가해야 함을 보여준다.

**키워드:** LLM evaluation, multi-turn robustness, sycophancy, answer survival, failure timing

## 1. 문제와 접근

LLM 평가는 흔히 단일 턴 정확도에 의존한다. 그러나 실제 사용 환경에서 대화는 한 번의 질의응답으로 끝나지 않는다. 사용자는 모델의 답을 반복해서 의심하거나, “전문가라면 다른 답을 고른다”와 같은 권위 주장을 제시하거나, 모델이 이전 답을 철회하도록 압박할 수 있다. 이때 중요한 질문은 모델이 처음부터 몰랐는지가 아니라, **처음에는 맞힌 답을 대화 중에도 유지할 수 있는가**이다.

이 문제는 기존 accuracy@1만으로는 포착하기 어렵다. 같은 초기 정확도를 가진 두 모델이라도, 하나는 첫 번째 반박에서 바로 오답으로 전향하고 다른 하나는 5라운드 동안 정답을 유지할 수 있다. 이는 사용자 압박·검증 질문이 반복되는 실제 배포 환경에서 중요한 안정성 차이다.

GALILEO는 이 차이를 측정하기 위해 ground-truth가 명확한 과제에서 초기 정답 사례만을 대상으로 persona pressure와 NRC를 같은 조건에서 비교한다. 이를 통해 단순 반복 질의로 인한 drift와 압박성 발화로 인한 answer flip을 분리하고, multi-turn robustness를 Survival@r, Fail@1/TTF, Recovery@flip으로 나누어 측정한다. 다만 본 2쪽 초안의 정량 headline은 Survival@5와 Fail@1에 집중한다.

## 2. 프로토콜과 실험 설정

GALILEO는 세 단계로 구성된다. Phase 1에서는 persona-free 초기 답변을 생성하고 task-specific evaluator로 정답 여부를 판정한다. 이후 robustness 지표는 초기 정답 사례에 조건부로 계산한다. Phase 2에서는 초기 정답 사례에 대해 R=5 라운드의 follow-up interaction을 수행한다. persona pressure 조건은 권위 주장, 강한 압박, 단순 부정, 논리 함정, 부드러운 압박 등 답변 변경을 유도할 수 있는 발화를 사용한다. NRC 조건은 같은 라운드 수와 유사한 응답 조건을 유지하되, 새로운 과제 관련 증거나 대안 답을 주지 않는 중립적 재확인 요청만 사용한다. Phase 3에서는 압박 단계에서 한 번이라도 flip된 사례에 대해 Recovery@flip을 측정할 수 있으나, 본 초안에서는 보조 지표로만 정의한다.

핵심 비교는 persona arm과 그에 대응하는 NRC를 **같은 initially-correct subset**에서 평가하는 persona-matched comparison이다. 따라서 Survival@5나 Fail@1의 차이는 서로 다른 분모에서 생긴 selection artifact가 아니라, 동일한 초기 정답 사례 위에서 pressure 조건과 control 조건을 비교한 값으로 해석한다. 다만 persona/task/seed별 matched subset이 다르므로 하나의 고정된 global effective \(n\)을 가정하지 않는다.

| 항목 | 내용 |
|---|---|
| 주요 모델 | Qwen2.5-7B-Instruct |
| Aggregated main effect | seed1–4 tracked Table W artifact, 5개 pressure persona의 persona-weighted pooled aggregate |
| 포함 과제 범위 | GSM8K, SVAMP, ARC-Easy, SQuAD 1.1/2.0, TriviaQA 계열 tracked artifact |
| Pressure persona | Authority Claim, Logical Trap, Simple Denial, Strong Pressure, Soft Pressure |
| Control | NRC: 같은 R=5 구조의 중립 재질문, 새 증거/대안 답 없음 |
| Effective N | headline CSV는 단일 global denominator를 노출하지 않음; persona/task/seed별 initially-correct subset 위에서 계산 |
| Case study | Qwen2.5-7B-Instruct seed1–3, authority_claim vs control_reask, GSM8K 및 ARC-Easy; launcher와 artifact naming 기준 dataset당 seed별 50-sample package |
| Evaluator | boxed final answer 추출 후 deterministic task-specific scorer 사용; math는 numeric/exact match, MCQA는 option-label exact match, QA는 normalized EM으로 binary correctness를 판정하고 F1은 보조 score로 기록 |
| 핵심 지표 | Survival@5, Fail@1 |
| 보조 지표 | Recovery@flip: 정의는 유지하되 headline 결과에서는 제외 |

Survival@r은 round r에서의 단순 정확도가 아니라, round 1부터 r까지 한 번도 실패하지 않은 누적 생존율이다. 따라서 중간에 오답으로 전향했다가 이후 다시 정답을 낸 사례는 Survival에는 실패로 기록되고, 회복 여부는 Recovery@flip에서 별도로 측정된다.

## 3. 결과

### 3.1 Persona pressure는 정답 유지율을 낮추고 초기 실패율을 높인다

Aggregated main effect에서 persona pressure는 NRC 대비 Survival@5를 낮추고 Fail@1을 높인다. 아래 표는 `table_w_effect_delta_seed1-4_20260209.csv`에 포함된 persona-weighted columns를 기준으로 한 compact headline이다. 다른 문서의 Table W 설명에는 delta-first macro 등 대체 집계 관례가 함께 남아 있으므로, 본 초안의 수치는 아래 CSV 열 정의에 한정해 해석한다.

| 지표 | NRC | Persona pressure | 차이 |
|---|---:|---:|---:|
| Survival@5 | 80.32% | 57.55% | **-22.76%p** |
| Fail@1 | 13.10% | 20.03% | **+6.93%p** |

이 결과는 R=5 horizon에서 persona pressure 조건의 누적 정답 유지율이 NRC보다 낮고, 첫 번째 압박 턴에서의 실패율은 더 높다는 것을 보여준다. 즉, 동일한 initially-correct subset 위에서 중립적 재확인만 반복한 조건보다 압박성 follow-up 조건에서 answer flip이 더 자주 관찰된다. Never-fail은 별도 artifact 항목으로 남아 있지만, 본 2쪽 초안에서는 headline 중복을 피하기 위해 별도로 보고하지 않는다.

### 3.2 Qwen2.5-7B-Instruct 사례에서는 ARC-Easy에서 감소폭이 더 크다

Qwen2.5-7B-Instruct seed1–3 authority-claim package에서는 GSM8K보다 ARC-Easy에서 Survival@5 감소 폭이 더 크게 관찰된다. 아래 표의 두 설정은 서로 다른 correction/recovery prompt package를 뜻하며, 별도의 모델 baseline을 뜻하지 않는다.

| Prompt package | Dataset | ΔSurvival@5 | 95% CI |
|---|---|---:|---:|
| evidence-bearing | GSM8K | **-0.215** | [-0.293, -0.167] |
| evidence-bearing | ARC-Easy | **-0.674** | [-0.755, -0.633] |
| grounded-correction | GSM8K | **-0.208** | [-0.341, -0.070] |
| grounded-correction | ARC-Easy | **-0.612** | [-0.653, -0.551] |

이 표는 현재 Qwen2.5-7B-Instruct tracked artifact 범위에서 과제별 압박 취약성이 다를 수 있음을 보여준다. 그러나 이는 단일 모델 계열과 두 데이터셋에 대한 관찰이므로, 객관식 또는 비수학형 과제가 일반적으로 더 취약하다고 단정하지 않는다. 더 넓은 일반화를 위해서는 추가 모델 family와 과제군에서 같은 persona-matched protocol을 반복해야 한다.

## 4. 해석과 한계

GALILEO의 NRC는 해석상 중요하다. multi-turn interaction 자체는 일반적인 drift를 만들 수 있으므로, persona pressure의 효과를 주장하려면 같은 라운드 수와 유사한 응답 조건을 가진 control이 필요하다. GALILEO는 persona arm과 NRC를 같은 initially-correct subset에서 비교함으로써 반복 대화의 일반 효과와 압박성 발화의 효과를 분리한다.

또한 Survival, Fail@1/TTF, Recovery@flip은 서로 다른 현상을 포착한다. Survival은 누적 정답 유지성을, Fail@1은 첫 반박에 대한 취약성을, Recovery는 오답 전향 이후 복구 가능성을 측정한다. 본 초안의 headline 결과는 Survival과 Fail@1에 집중하지만, recovery 축을 별도로 정의하는 것은 “끝까지 버티는 능력”과 “무너진 뒤 돌아오는 능력”을 혼동하지 않기 위해 필요하다.

한계도 명확하다. 첫째, 가장 강한 정량 주장은 현재 tracked artifact, 특히 Qwen2.5-7B-Instruct와 일부 aggregated artifact에 기반한다. 둘째, 본문 case study는 GSM8K와 ARC-Easy 두 데이터셋에 집중하므로 과제 형식에 따른 취약성 차이를 일반 법칙으로 확대 해석할 수 없다. 셋째, 모든 robustness 지표는 초기 정답 사례에 조건부로 계산되므로 전체 accuracy가 아니라 “초기 성공 이후의 deviation dynamics”를 설명한다. 넷째, 본 프로토콜은 ground-truth가 있는 과제에 초점을 두며, 주관적 판단이나 장문 생성 과제에는 별도 평가 설계가 필요하다.

## 5. 결론

GALILEO는 LLM의 multi-turn correctness를 단일 정확도가 아니라 시간에 따른 정답 유지와 초기 실패 취약성으로 측정하는 drift-controlled 평가 프로토콜이다. 현재 tracked artifact는 persona pressure가 NRC 대비 Survival@5를 낮추고 Fail@1을 높인다는 근거를 제공한다. 따라서 LLM의 실제 대화형 신뢰성을 평가하려면 초기 정답률뿐 아니라, 압박 속에서의 survival trajectory와 early-failure behavior를 함께 보고해야 한다.

본문 수치는 `docs/paper/artifacts/table_w_effect_delta_seed1-4_20260209.csv`, `docs/paper/artifacts/qwen7b_evidence_multiseed_seed1-3_deltas_20260310.csv`, `docs/paper/artifacts/qwen7b_grounded_multiseed_seed1-3_deltas_20260310.csv`에서 가져왔다. 프로토콜과 지표 정의는 `docs/paper/FIGURE_CAPTIONS.md`, `docs/paper/CLAIM_EVIDENCE_MAP.md`, `docs/paper/PAPER_EVIDENCE_STATUS_20260310.md`를 따른다.
