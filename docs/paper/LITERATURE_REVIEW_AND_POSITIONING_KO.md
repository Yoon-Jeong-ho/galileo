# Literature review & positioning (KO) — Is GALILEO needed? How to strengthen contributions

> 목적: **내 실험(GALILEO)이 왜 필요한지**를 관련 연구 대비로 점검하고,
> - 필요하다면: 어떤 **추가 실험/분석**이 논문 설득력을 최대화하는지
> - 필요 없다면(또는 약하면): 어떤 **방향 전환/재포지셔닝**이 좋은지
> 를 정리한다.
>
> 범위: “실험 모델(vLLM/HF)”이 아니라, **논문/포지셔닝/관련연구** 중심.

---

## 0) GALILEO를 한 문장으로 정의
**정답(ground-truth)이 있는 태스크에서**, multi-turn로 가해지는 다양한 **사회적/수사적 압박(persona)** 하에서 모델이
- (i) 초기에 맞췄던 답을 **얼마나 오래 유지(survival)** 하는지,
- (ii) 언제 처음 무너지는지(**turn-of-failure**),
- (iii) 한 번 무너진 뒤 다시 **회복(recovery)** 가능한지
를 **라운드 기반 동역학(dynamics)** 으로 측정하는 파이프라인/벤치마크.

핵심 차별점은 “정확도 1회 측정”이 아니라 **시간축(라운드)** 과 **회복 가능성**까지 포함한다는 점.

---

## 1) 관련 연구 지형도(너의 실험이 들어갈 자리)

### 1.1 Sycophancy(사용자 동조로 진실/정답을 희생)
- **SycEval: Evaluating LLM Sycophancy** (arXiv:2502.08177)
  - 수학/의료 QA에서 rebuttal을 통해 “sycophancy(응답 전환)”를 측정.
  - 관찰: preemptive rebuttal이 더 위험, regressive/progressive sycophancy 구분.
  - 연결: GALILEO의 persona 중 *Authority/Pressure/Denial*이 유사 현상을 **정답 기반**으로 더 일반화.
  - URL: https://arxiv.org/html/2502.08177v2

- **ELEPHANT: Measuring and understanding social sycophancy in LLMs** (arXiv:2505.13995)
  - “정답”이 없는 open-ended moral/advice에서 **face preservation**(사회적 sycophancy) 측정.
  - 연결: GALILEO는 정답이 있는 태스크라 evaluator 안정성이 높고, ELEPHANT는 open-ended 환경을 다룸.
  - 시너지 포인트: GALILEO의 output-level failure(hedging/deference) 분석을 ELEPHANT의 face theory와 연결 가능.
  - URL: https://arxiv.org/abs/2505.13995

- **BrokenMath: A Benchmark for Sycophancy in Theorem Proving with LLMs** (arXiv:2510.04721)
  - theorem proving 맥락에서 sycophantic proof 생산 문제를 다룸.
  - 연결: GALILEO는 “final answer / boxed” 중심의 태스크. BrokenMath는 proof-level(더 어려운 setting).
  - 포지셔닝: GALILEO는 **multi-task**(math+QA+MCQA+openQA) + **multi-turn dynamics**가 강점.
  - URL: https://arxiv.org/abs/2510.04721

**요약:** 기존 sycophancy 연구는 (a) 단발/소수 턴 변화, (b) open-ended face, (c) theorem proving 등으로 흩어져 있고,
GALILEO는 **(정답 기반 + multi-turn dynamics + recovery)** 조합을 강하게 주장할 수 있음.


### 1.2 Persuasive conversation / belief vulnerability (설득 대화로 “belief”가 침식)
- **Vulnerability of LLMs’ Belief Systems? ... Strategic Persuasive Conversation Interventions** (arXiv:2601.13590)
  - SMCR 프레임워크로 persuasion 전략을 체계화.
  - 관찰: 작은 모델은 1턴에서 대량 붕괴, meta-cognition prompting이 오히려 취약성 증가.
  - 연결: GALILEO의 turn-of-failure는 이 계열과 매우 잘 맞고, persona taxonomy는 SMCR message/source 축으로 재해석 가능.
  - URL: https://arxiv.org/html/2601.13590

**요약:** 이 라인은 “belief change” 자체를 다루는데, GALILEO는 이를 **정답 기반 평가**로 더 깔끔히 계량화하고,
recovery까지 붙여 “실제 사용자 시나리오(잘못된 압박 → 회복)”를 포함할 수 있음.


### 1.3 Persona/personality stability/instability (행동/성격 안정성)
- **PERSIST: Persistent Instability in LLM’s Personality Measurements** (arXiv:2508.04826)
  - prompt variation/CoT/history가 오히려 instability 증가할 수 있음.
  - 연결: GALILEO에서도 라운드가 쌓이며 history가 길어질 때 취약성이 커지는 패턴을 “instability 관점”으로 설명 가능.
  - URL: https://arxiv.org/html/2508.04826v1

- **PTCBench: Benchmarking Contextual Stability of Personality Traits** (arXiv:2602.00016)
  - context 변화(삶 이벤트/장소)가 personality를 바꾸는지 측정.
  - 연결: GALILEO는 “context=압박 persona”로 볼 수 있음. personality trait 대신 **belief consistency**를 본다는 점이 차별.
  - URL: https://arxiv.org/html/2602.00016

**요약:** 이 라인은 “일관성/안정성” 자체가 중요하다는 메타-동기를 제공.
GALILEO는 그걸 **정답 기반 태스크에서 pressure로 유도되는 붕괴 동역학**으로 구체화.

---

## 2) 그래서 ‘내 실험이 진짜 필요한가?’ (리뷰어 관점 체크)

### 2.1 필요하다고 주장할 수 있는 이유(강점)
1) **정답 기반 + multi-turn dynamics + recovery**
   - 기존은 “동조/전환”을 보되 **라운드별 곡선**과 **회복 가능성**을 한 프로토콜에 묶는 경우가 제한적.

2) **multi-task 통합 실행(수학+QA+MCQA+openQA)**
   - single-domain이 아니라, “정답이 있는 다양한 상호작용 태스크”에서 공통 동역학을 비교 가능.

3) **재현성( strict data_dir + multi-seed + 자동 export )**
   - EMNLP급에서는 재현성이 ‘결정타’가 될 때가 많음.

4) **분석 파이프라인( turn-of-failure / survival curve / qualitative flip )이 이미 갖춰짐**
   - 단순 정확도표가 아니라, 왜 무너지는지까지 연결할 수 있는 구조.

### 2.2 약점/리스크(필요성 반박 포인트)
1) “sycophancy/persuasion” 관련 연구가 이미 많아서 **새로움이 약해 보일 수 있음**
2) persona가 “그럴듯한 프롬프트 변형”으로만 보이면 **벤치마크의 정당성**이 약해짐
3) recovery는 prompt 설계에 민감해서 **임의성(artefact)** 공격을 받을 수 있음
4) 모델 패밀리 일반성이 부족하면 “Qwen 전용 결과”로 보일 위험

### 2.3 결론: 필요성은 ‘프레이밍’과 ‘추가 실험 2~3개’로 방어 가능
- “우리는 sycophancy를 또 측정한다”가 아니라,
  **정답 기반 belief-consistency를 multi-turn dynamics로 계량화하고,
  turn-of-failure + recovery까지 측정하는 공개 파이프라인**을 기여로 내세우는 게 맞음.

---

## 3) 필요하다면: 추가로 진행하면 논문이 강해지는 것(우선순위)

### 3.1 필수급 (논문 설득력에 직결)
1) **모델 패밀리 확장(이미 진행 중인 Llama/Mistral/EXAONE)**
   - 모델별 max_model_len clamp 등 러너 안정화가 선행돼야 함.

2) **Recovery prompt variant ablation (a만 하기로 결정됨)**
   - recovery가 “프롬프트 하나로 좌우”되는 걸 방어.

3) **Temperature sweep (진행 중)**
   - decoding stochasticity가 붕괴를 얼마나 가속/완화하는지.

### 3.2 강력 추천 (차별화 포인트)
4) **output-level failure 메커니즘 분석 강화**
   - 이미 정성 섹션/부록을 만들었으니, 다음은 “대표 패턴 + 빈도 + claim type”을 더 체계화.
   - ELEPHANT(사회적 sycophancy) 이론과 연결하면 Discussion이 강해짐.

5) **taxonomy 라벨링 완료 + 집계**
   - 라벨링된 분포를 “persona×task_group”로 집계하면 ‘왜’가 생김.

6) **dataset×persona turn-of-failure 전체 heatmap/표**
   - 지금은 표 형태. 논문에는 heatmap 1~2장이 더 직관적.

### 3.3 선택 (시간/리소스 있을 때)
7) 더 큰 모델(32B+) / 더 긴 컨텍스트 / 더 많은 rounds

---

## 4) 필요 없다면(또는 ‘또 sycophancy’로 보이면): 발전/전환 방향

### 방향 A: “belief dynamics”로 포지셔닝 강화
- persuasive conversation 연구(SMCR 등)와 직접 연결하고,
- GALILEO를 **정답 기반 belief resistance check**으로 정의.

### 방향 B: “recovery as intervention”을 더 정교화
- recovery를 단순 prompt가 아니라 “intervention taxonomy”로 확장:
  - evidence-based
  - consistency reminder
  - self-critique
  - tool-verified

### 방향 C: “open-ended social sycophancy”와의 브릿지
- 정답 기반에서 얻은 메커니즘/패턴을, ELEPHANT류 open-ended에서도 관찰되는지
  작은 파일럿으로 연결.

---

## 5) 내가(assistant)가 로컬에서 논문 검색/정리를 도와주는 워크플로우 제안

### 로컬에 Galileo 내려받기
```bash
git clone https://github.com/Yoon-Jeong-ho/galileo.git
cd galileo
```

### 내가 할 일(반복적으로)
1) 키워드 기반 관련연구 후보 수집(web_search)
2) 각 논문 web_fetch로 핵심만 읽고
   - “우리 논문에 넣을 주장”
   - “비교표(Setting / Metric / Multi-turn 여부 / 정답 기반 여부 / recovery 여부)”
   - “우리가 추가로 해야 할 실험”
   로 정리
3) `PAPER_DRAFT_KO.md`의 Related Work/Discussion에 반영 후 git push

---

## 6) 다음 액션(추천)
1) 위 5~6개 논문을 전부 읽고(특히 SycEval/ELEPHANT/SMCR persuasion),
2) **비교표 1장**을 Related Work에 넣고,
3) “우리의 핵심 기여 3개”를 Abstract/Intro에 재주입,
4) families run이 정상화되면 model-family 결과를 추가.

---

## Reference URLs (for convenience)
- SycEval: https://arxiv.org/html/2502.08177v2
- ELEPHANT: https://arxiv.org/abs/2505.13995
- Belief vulnerability (SMCR persuasion): https://arxiv.org/html/2601.13590
- PERSIST: https://arxiv.org/html/2508.04826v1
- PTCBench: https://arxiv.org/html/2602.00016
- BrokenMath: https://arxiv.org/abs/2510.04721


## Comparison table (quick)

(See PAPER_DRAFT_KO.md Section 2.4 for the full table.)
