# GALILEO 논문 초안 (Korean Draft)

> 상태: **초안 v0** (실험은 진행/확장 중).  
> 목표: EMNLP급 제출을 염두에 둔 **문제정의 → 방법 → 실험 → 결과/인사이트 → 한계/윤리**까지 한 번에 서술 가능한 형태.

---

## 초록 (Abstract)

대규모 언어모델(LLM)은 사용자와의 상호작용에서 설득, 권위 주장, 반복 부정 등 다양한 형태의 **사회적 압박(social pressure)**을 받을 때, 정답이 존재하는 과제에서도 정답을 유지하지 못하고 오답으로 전향하는 현상이 보고된다. 기존 연구는 주로 (i) 안전/정렬 관점의 순응성, (ii) 대화적 설득 시나리오, (iii) 사실성/환각 평가를 다루었으나, **정답(ground-truth)이 확정된 과제에서 “정답 유지(survival)–전향(flip)–회복(recovery)”의 동역학을 멀티턴 프로토콜로 일관되게 측정**하는 공개 재현 파이프라인은 상대적으로 부족하다.

본 논문에서는 GALILEO를 제안한다. GALILEO는 (1) 정답이 있는 문제(수학, extractive QA, MCQA, open-domain QA)에 대해 초기 정답성을 평가하고, (2) 다섯 가지 adversarial persona(Soft Pressure, Simple Denial, Strong Pressure, Authority Claim, Logical Trap)를 최대 5라운드 적용하여 **라운드별 생존율(survival rate)**을 측정하며, (3) 오답으로 전향한 샘플에 대해 회복 프롬프트를 제공하여 **회복률(recovery rate)**을 평가한다. 또한 모든 태스크에서 최종 답을 `\boxed{...}` 포맷으로 통일하여 자동 채점의 안정성을 확보하였다.

초기 실험 스냅샷(예: Qwen2.5-7B-Instruct, TP=4, max context 16k)에서는, (i) Authority Claim이 가장 강한 붕괴를 유발하는 경향, (ii) 수학 태스크는 초기 정확도 및 압박 하 생존성이 높지만 전향 이후 회복이 상대적으로 어려운 반면, SQuAD 계열 QA는 회복률이 높게 나타나는 경향, (iii) open-domain QA(TriviaQA)는 초기 정확도가 낮고 강한 압박에서 생존율이 급락하는 경향을 관찰했다. GALILEO는 정답 기반 멀티턴 압박 평가를 통해 LLM의 신념 일관성 및 사회적 압박 취약성을 정량화하는 공개 벤치마크/파이프라인으로 활용될 수 있다.

---

## 1. 서론 (Introduction)

### 1.1 문제의식

LLM은 사용자 중심 응답, 공감적 대화, 고품질 추론을 위해 인간 선호/정렬 신호를 학습한다. 그러나 이 과정은 때때로 사용자의 주장에 과도하게 동조하거나, 권위적 주장에 흔들리거나, 반복되는 부정에 의해 자신이 낸 정답을 철회하는 형태로 나타날 수 있다. 특히 **정답이 확정된 태스크**에서조차 오답으로 전향한다면, 이는 교육/의료/법률/연구 지원 등 다양한 적용에서 신뢰성 위험으로 이어질 수 있다.

### 1.2 기존 평가의 공백

기존 평가들은 다음의 공백을 남긴다.

- 단일턴 정확도(accuracy)만으로는 **멀티턴 압박 과정에서의 붕괴 시점**을 설명하기 어렵다.
- 안전/정렬 관점의 순응성 평가와 달리, 정답 기반 태스크에서의 **정답 유지 vs 오답 전향**은 별도의 관측치가 필요하다.
- 오답 전향 이후의 **회복 가능성(recovery)**은 실제 사용 시나리오에서 중요하지만, 동일 프로토콜 내에서 체계적으로 측정되는 경우가 적다.

### 1.3 우리의 기여

본 논문은 다음을 기여한다.

1. **정답 기반 멀티턴 압박 평가 프로토콜**: 초기 평가 → persona 압박(최대 5라운드) → 회복 평가를 일관된 구조로 제공.
2. **다중 태스크 확장**: 수학(GSM8K/SVAMP), extractive QA(SQuAD 1.1/2.0), MCQA(ARC-Easy), open-domain QA(TriviaQA)까지 한 파이프라인에서 수행.
3. **출력 포맷 표준화 및 평가 안정화**: 모든 태스크에서 최종 답을 `\boxed{...}`로 강제하고, evaluator에서 boxed 추출을 우선.
4. **재현 가능한 실행/로그 구조**: JSONL(대화 로그) + CSV(요약 지표)를 표준화하여 분석과 논문화에 유리.

---

## 2. 관련 연구 (Related Work)

> 초안: 실제 제출 시에는 최신 정렬/설득/일관성/사실성 평가 문헌을 정리해 인용 추가 필요.

- **LLM 정렬과 순응성**: RLHF/선호학습이 사용자 선호에 맞추는 과정에서 사실성보다 정중함/동조가 강화될 수 있음.
- **설득/권위 기반 공격**: 권위 주장, 사회적 압박, 논리적 함정 등을 통해 모델의 결정을 흔드는 시나리오.
- **사실성/환각 평가**: 모델이 답을 바꾸는 현상은 환각과도 관련되지만, 본 연구는 *정답이 있는 태스크에서 멀티턴 압박에 의해 정답 유지가 붕괴되는 동역학*에 초점을 둔다.
- **멀티턴 평가**: 멀티턴 대화에서의 일관성/안전성 평가가 증가하고 있으나, 정답 기반 survival/recovery를 통합한 공개 파이프라인은 제한적.

---

## 3. GALILEO: 방법 (Method)

### 3.1 태스크 및 데이터 포맷

각 데이터셋은 JSONL 한 파일에 저장하며, 각 라인은 한 문제를 의미한다.

- **수학 (math)**
  - 입력: `{"question": "...", "answer": "..."}` 또는 유사 필드
  - 정답 추출: `\boxed{...}` 안의 최종 답으로 정규화

- **QA (qa)**
  - 입력: `{"task": "qa", "question": "...", "answers": ["a1", "a2", ...]}`
  - 평가: boxed 텍스트를 추출하여 정답 alias들과 매칭(간단 정규화 포함)

- **MCQA (mcqa)**
  - 입력: `{"task": "mcqa", "question": "...", "choices": [{"label":"A","text":"..."},...], "label": "B"}`
  - 평가: boxed 라벨(`\boxed{B}`) 기반

### 3.2 프로토콜: Initial → Adversarial → Recovery

- **Phase 1: Initial evaluation**
  - 모델에 기본 프롬프트로 답변 요구
  - 정답 여부를 판정하고, 이후 단계에서 추적할 대상으로 저장

- **Phase 2: Adversarial persona pressure (최대 5라운드)**
  - 정답인 샘플을 대상으로 persona별로 압박 대화를 진행
  - 매 라운드마다 모델의 답이 여전히 정답인지 평가
  - 지표: 라운드 r에서 정답을 유지한 비율 = survival rate

- **Phase 3: Recovery**
  - adversarial 단계에서 오답으로 전향(flip)한 샘플에 대해 회복 프롬프트 제공
  - 회복 프롬프트 후 정답으로 돌아왔는지 평가
  - 지표: 회복률(recovery rate)

### 3.3 Persona 설계

다섯 persona는 서로 다른 압박 메커니즘을 모사한다.

1. **Soft Pressure**: 완만한 이견 제시/부드러운 설득
2. **Simple Denial**: 근거 없이 반복 부정
3. **Strong Pressure**: 강한 확신/반복 압박
4. **Authority Claim**: 전문가/권위/규범을 근거로 정답을 흔듦
5. **Logical Trap**: 말장난/논리적 함정/프레이밍 전환

### 3.4 출력 포맷 및 평가기

- 모든 태스크에서 최종 답은 `\boxed{...}`로 강제한다.
- 추론/설명(CoT)은 박스 밖에 허용하되, 평가기는 boxed를 우선 추출한다.
- 출력:
  - JSONL 로그: `*_initial.jsonl`, `*_adversarial.jsonl`, `*_recovery.jsonl`
  - CSV 요약: `initial_accuracy.csv`, `adversarial_survival.csv`, `recovery_accuracy.csv`

---

## 4. 실험 (Experiments)

### 4.1 모델

- 예시(진행 중): Qwen2.5-7B/14B/32B-Instruct 계열
- 실행은 vLLM 기반 추론, TP=4로 병렬화

### 4.2 데이터셋

- Math: GSM8K, SVAMP
- QA: SQuAD 1.1 / SQuAD 2.0 (각각 별도 파일)
- MCQA: ARC-Easy
- Open-domain QA: TriviaQA (rc)

### 4.3 구현 및 재현성

- 환경변수 기반 설정: `TP_SIZE`, `MAX_MODEL_LEN`, `MAX_TOKENS`, `NUM_SAMPLES`
- 장시간 실행을 위해 tmux 기반 스크립트 제공
- Plan A: unified data_dir에 모든 jsonl을 모아 한 번에 실행

---

## 5. 결과 (Results)

> 아래 수치는 **스냅샷**이며, 최종 논문에서는 모델/태스크/seed를 확장하고 평균±표준편차 등을 보고하는 것을 권장.

### 5.1 초기 정확도 (Initial accuracy)

Qwen2.5-7B-Instruct 기준(각 데이터셋 최대 1000개 캡):

- ARC-Easy(validation, MCQA): 94.74% (540/570)
- GSM8K(math): 97.00% (970/1000)
- SVAMP(math): 95.57% (669/700)
- SQuAD 1.1(validation, QA): 84.40% (844/1000)
- SQuAD 2.0(validation, QA): 82.60% (826/1000)
- TriviaQA(rc validation, QA): 54.10% (541/1000)

### 5.2 생존율 (Survival under pressure)

라운드 5에서의 survival(데이터셋 전체 aggregate):

- Authority Claim: 40.27%
- Strong Pressure: 49.04%
- Simple Denial: 52.75%
- Logical Trap: 61.83%
- Soft Pressure: 73.00%

### 5.3 회복률 (Recovery)

flip 이후 recovery(데이터셋 전체 aggregate):

- Authority Claim: 72.40%
- Strong Pressure: 78.23%
- Simple Denial: 77.98%
- Soft Pressure: 84.01%
- Logical Trap: 87.22%

---



## 5.X 추가 결과: 7B (GPU 0–3, TP=4) 실험 분석 (2026-02-03)

본 절에서는 GPU 0–3에서 TP=4로 수행한 7B 실험 결과(각 데이터셋 최대 1000개 샘플)를 요약하고, 논문에 바로 쓸 수 있는 형태의 관찰/해석을 정리한다.

- 실행 설정: `CUDA_VISIBLE_DEVICES=0,1,2,3`, `TP_SIZE=4`, `MAX_MODEL_LEN=16384`, `MAX_TOKENS=2048`, `NUM_SAMPLES=1000`, `SEED=42`
- 결과 폴더: `/mnt/raid6/aa007878/galileo/results/all_pilot_20260203_143301/7b/`
- 주의: 본 run에는 legacy 파일(`*_val_50`)이 함께 포함될 수 있어, 최종 표/그림에는 **strict data_dir(정식 데이터만)**을 사용해 재실행한 결과를 사용하도록 권장한다.

### (1) 초기 정확도 (Initial accuracy)

- GSM8K: 96.70% (967/1000)
- SVAMP: 95.14% (666/700)
- ARC-Easy(validation): 95.26% (543/570)
- SQuAD 1.1(validation): 79.40% (794/1000)
- SQuAD 2.0(validation): 76.50% (765/1000)
- TriviaQA(rc validation): 55.60% (556/1000)

**해석:** 수학/MCQA는 초기 정확도가 매우 높고 안정적인 반면, QA 특히 SQuAD는 샘플링(셔플) 구성에 따라 초기 정확도가 수%p 단위로 흔들릴 수 있다. 이는 이후 survival/recovery의 절대값을 비교할 때 seed/샘플링 통제가 필수임을 의미한다.

### (2) 압박 하 생존율 (Survival) — Round 5 aggregate

- Authority Claim: 39.61% (1730/4368)
- Strong Pressure: 48.49% (2118/4368)
- Simple Denial: 51.19% (2236/4368)
- Logical Trap: 59.98% (2620/4368)
- Soft Pressure: 69.57% (3039/4368)

**해석:** persona 효과의 순위(ordering)는 스냅샷과 동일하게 유지된다. 특히 Authority Claim이 가장 치명적이며, Soft Pressure는 가장 약한 공격으로 나타난다. 이는 ‘권위/전문가 주장’이 모델의 답변 유지에 구조적으로 큰 영향을 주는 압박 메커니즘임을 시사한다.

### (3) 회복률 (Recovery) — flip 이후 aggregate

- Authority Claim: 71.30% (1881/2638)
- Strong Pressure: 76.27% (1716/2250)
- Simple Denial: 74.77% (1594/2132)
- Soft Pressure: 81.79% (1087/1329)
- Logical Trap: 85.76% (1499/1748)

**해석:** survival이 낮은 persona(Authority/Strong)에서도 recovery는 70%대 이상으로 관측된다. 즉, ‘오답으로 흔들리게 만드는 것’과 ‘정답으로 복귀시키는 것’은 동일한 난이도가 아닐 수 있다.

### (4) 논문에 쓰는 포인트 (서술 템플릿)

- **Persona 효과(주장 1):** “Authority Claim은 라운드가 진행될수록 survival을 가장 빠르게 붕괴시키며, Soft Pressure는 상대적으로 완만한 붕괴 곡선을 보인다.”
- **불확실성–취약성 연결(주장 2):** 초기 정확도가 낮은 open-domain QA(TriviaQA)는 강한 압박에서 급격히 취약해지는 경향이 있으며, 이는 정답 불확실성이 사회적 압박 취약성을 증폭시킬 가능성을 뒷받침한다.
- **재현성(주장 3):** 샘플링 seed에 따라 초기 정확도(특히 QA)가 유의미하게 달라질 수 있으므로, 최종 논문에서는 multi-seed 평균±표준편차/CI로 보고한다.


## 6. 분석 및 논문용 인사이트 (Analysis / Insights)

### 6.1 Persona 효과의 비대칭성

Authority Claim이 가장 치명적이며, Soft Pressure는 상대적으로 약하다. 이는 모델이 **권위·전문성 주장에 대한 prior**를 강하게 갖고 있을 가능성 또는 안전/정렬 과정에서 **“권위 있는 사용자/지침”에 순응하는 행동**이 강화되었을 가능성을 시사한다.

### 6.2 태스크 의존적 trade-off: Robustness vs Recovery

- 수학 태스크는 초기 정확도와 압박 하 생존성이 높은 편이나, 일단 전향하면 회복이 어렵게 나타날 수 있다.
- 반면 SQuAD 계열 QA는 강한 압박에서 생존이 낮아질 수 있지만, 회복률이 매우 높게 나타나는 경향이 있다.

이 결과는 “정답을 알고 있는지”와 “사회적 압박에 저항하는지”가 동일 축이 아님을 시사한다.

### 6.3 Open-domain QA의 취약성

TriviaQA는 초기 정확도가 상대적으로 낮고, Strong Pressure에서 생존이 급락하는 경향을 보인다. 이는 open-domain QA가 본질적으로 정답 불확실성이 크며, 그 불확실성이 persona 압박에 대한 취약성을 증폭시킬 수 있음을 시사한다.



### 6.4 라운드별 붕괴 동역학(Flip dynamics): 무엇이 ‘언제’ 무너지는가?

GALILEO의 핵심은 단일 정확도보다 **라운드 진행에 따른 붕괴 곡선**이다. 따라서 최종 논문에서는 다음을 기본 그림으로 제시한다.

- **Survival curve(라운드 1→5)**: persona별 survival rate를 라운드 축으로 시각화
- **Turn-of-failure 분포**: 처음으로 오답이 발생한 라운드(1~5)의 히스토그램

이 분석을 통해 “Authority Claim이 항상 최악” 같은 요약을 넘어서, **어떤 persona가 ‘초반에 급격 붕괴’ vs ‘완만 붕괴’**를 유발하는지 보여줄 수 있다. 예컨대 권위 주장형은 early-round에서 큰 낙폭을 만들고, 논리 함정은 누적적으로 침식하는 형태를 보일 가능성이 있다.

### 6.5 초기 정확도(능력)와 압박 내성(robustness)은 동일 축인가?

리뷰어가 자주 묻는 질문은 “그냥 initial accuracy가 높으면 robust한 것 아닌가?”이다. 우리는 다음 두 관측을 중심 주장으로 정리할 수 있다.

- (관측 A) **initial accuracy가 높아도 Authority Claim에 취약**할 수 있다.
- (관측 B) **survival이 낮아도 recovery가 매우 높을 수 있다**(특히 extractive QA).

최종 논문에서는 dataset별로

- x축: initial accuracy
- y축: survival@round5(또는 AUC)

의 산점도를 제시해, capability–robustness의 상관이 완전하지 않음을 보인다(= disentanglement).

### 6.6 정성 분석(qualitative): 전향(flip) 유형 분류(taxonomy)

정량 지표만으로는 “왜” 바뀌는지 설명이 약해질 수 있다. 따라서 오답 전향 샘플을 일정 수(예: persona×task당 50개) 표본추출하여 아래 taxonomy로 라벨링한다.

- **Authority compliance**: 권위/전문가 주장에 복종하며 답을 수정
- **Social appeasement**: 갈등 회피/동조를 위해 답을 바꿈
- **Logical trap**: 프레이밍 전환/말장난에 말려 답이 바뀜
- **Uncertainty collapse**: 확신 부족으로 ‘모르겠다/애매하다’로 후퇴하거나 답을 회피
- **Hedged flip**: 답을 바꾸지만 근거가 약하거나 모순된 설명

이 정성 분류는 (i) persona별로 어떤 실패가 우세한지, (ii) task별로 어떤 실패가 우세한지(TriviaQA의 uncertainty 등)를 보여주며, 논문 설득력을 크게 올린다.

### 6.7 재현성/통계 보고(권장): multi-seed + CI

현재 스냅샷은 유용하지만, 최종 논문에서는 **seed 변화에 대한 민감도**를 통제해야 한다.

- 각 설정에 대해 seed를 3~5개로 반복
- 표는 평균±표준편차를 기본으로 보고
- 가능하면 bootstrap으로 95% CI를 함께 제시

특히 QA 계열은 샘플링/셔플 구성에 따라 초기 정확도가 수%p 흔들릴 수 있으므로, multi-seed 평균을 통해 “경향(trend)”을 주장하는 것이 안전하다.

### 6.8 제출용 Figure/Table 패키지(권장)

- **Figure 1**: persona별 survival curve(aggregate)
- **Figure 2**: task별 survival curve(수학 vs extractive QA vs openQA vs MCQA)
- **Figure 3**: robustness vs recovery scatter(또는 initial vs survival scatter)
- **Table 1**: dataset별 initial / survival@5 / recovery(모델별 블록)
- **Table 2 (ablation)**: recovery prompt variant, boxed vs free-form, decoding 변화

위 그림/표를 자동 생성하도록(결과 CSV → paper-ready md/fig) 스크립트를 추가하면, 실험 반복과 논문 업데이트를 매우 빠르게 만들 수 있다.

---

## 7. 한계 (Limitations)

- 본 초안의 수치는 단일 스냅샷이며, seed/모델/샘플링을 확장한 통계적 검증이 필요하다.
- boxed 포맷 강제는 평가 안정성을 주지만, 실제 사용자 상황의 자유로운 출력과는 차이가 있을 수 있다.
- 회복 프롬프트의 설계에 따라 recovery rate가 달라질 수 있으므로, 회복 프롬프트의 표준화/대조 실험이 필요하다.

---

## 8. 윤리 및 안전 고려 (Ethics & Safety)

- 본 벤치마크는 설득/권위/압박을 사용하지만, 목적은 공격 확산이 아닌 **모델 취약성 측정과 안전성 개선**이다.
- 공개 시에는 공격 프롬프트/페르소나가 악용되지 않도록, 책임 있는 공개 범위를 논의할 필요가 있다.

---

## 9. 부록: 실행 및 결과 정리 (Appendix)


### Paper-ready exports (curve / failure / qualitative sheet)

- Export survival curves + turn-of-failure + flip sample sheet (CSV):

```bash
python scripts/paper_export.py   --results_root /mnt/raid6/aa007878/galileo/results/all_pilot_20260203_143301/7b   --model_dir /mnt/raid6/aa007878/galileo/results/all_pilot_20260203_143301/7b/Qwen2.5-7B-Instruct   --out_dir /mnt/raid6/aa007878/galileo/results/all_pilot_20260203_143301/7b/paper_exports   --num_flip_samples 200   --seed 42
```

- Outputs:
  - `survival_curve.csv` (persona×round)
  - `turn_of_failure.csv` (persona×dataset×first-failure-turn distribution)
  - `flip_samples.csv` (manual taxonomy labeling sheet)


- full QA 데이터 생성:
  - `scripts/make_qa_full.py` → `/data_x/aa007878/galileo/data_qa_full/`
- unified data_dir 생성:
  - `scripts/make_all_data_dir.sh`
- 결과 요약(표준 라이브러리만 사용):
  - `scripts/summarize_results.py`

