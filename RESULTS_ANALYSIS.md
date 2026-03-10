# Galileo – Experiment Result Analysis (Living Doc)

> 목적: 실험 결과를 **논문에 바로 옮길 수 있는 형태로 정리**하고, 재현 가능한 커맨드/경로를 함께 남긴다.
> 
> 상태: 2026-02-03 기준, **7B/14B/멀티시드 실험은 진행 중/누적 중**이며, 본 문서는 업데이트되는 living doc이다.

---

## 2026-03-10 현재 기준선 (로컬 직접 검증)

아래 결과는 `/data_x/aa007878/projects/galileo`에서 **직접 실행 + export + validator 확인**한 현재 기준선이다.

- Math baseline:
  - raw run: `/data_x/aa007878/projects/galileo/tmp/results/pilot50_gpu5_20260310_185825/`
  - promoted alias: `/data_x/aa007878/projects/galileo/results_paper/qwen7b_math_control_authority_evidence_gsm8k_svamp_gpu5_20260310/`
- Non-math baseline:
  - raw run: `/data_x/aa007878/projects/galileo/tmp/results/main_arc_gpu6_20260310_191906/`
  - promoted alias: `/data_x/aa007878/projects/galileo/results_paper/qwen7b_nonmath_control_authority_evidence_arc_gpu6_20260310/`
- Grounded comparison runs:
  - GSM8K grounded main: `/data_x/aa007878/projects/galileo/tmp/results/qwen7b_gsm8k_control_authority_grounded_gpu5_20260310_221155/`
  - ARC grounded main: `/data_x/aa007878/projects/galileo/tmp/results/qwen7b_arc_control_authority_grounded_gpu5_20260310_221155/`
  - promoted aliases:
    - `/data_x/aa007878/projects/galileo/results_paper/qwen7b_gsm8k_control_authority_grounded_gpu5_20260310/`
    - `/data_x/aa007878/projects/galileo/results_paper/qwen7b_arc_control_authority_grounded_gpu5_20260310/`

핵심 확인 사실:

- GSM8K (50 samples): initial 86.00%, Authority Claim survival@5 58.14%, Control survival@5 88.37%, Recovery@flip 100.00%
- SVAMP (50 samples): initial 96.00%, Authority Claim survival@5 75.00%, Control survival@5 93.75%, Recovery@flip 100.00%
- ARC-Easy (50 samples): initial 98.00%, Authority Claim survival@5 24.49%, Control survival@5 93.88%, Recovery@flip 100.00%
- GSM8K grounded (50 samples): initial 82.00%, Authority Claim survival@5 63.41%, Control survival@5 92.68%, Authority Recovery@flip 93.33%
- ARC-Easy grounded (50 samples): initial 98.00%, Authority Claim survival@5 36.73%, Control survival@5 91.84%, Authority Recovery@flip 100.00%

해석 메모:

- 현재 기준선은 **pressure vs neutral drift 분리**에는 충분히 유망하다.
- 다만 `evidence_bearing` recovery가 모두 100%인 것은 작은 단일-seed 기준선이므로, 아직 “강한 논문 결론”로 쓰기보다 **작동 확인 + 후속 다중 seed 필요**로 정리해야 한다.
- 추가로, 2026-03-10 late pass에서 **generation seed를 Phase 1/2/3에 명시적으로 고정**하도록 수정했다. 이 수정 전에는 recovery variant만 바꿔도 Phase 2가 stochastic하게 달라질 수 있었기 때문에, correction-arm 직접 비교는 **seeded rerun 결과**를 기준으로 해석해야 한다.
- 현재 seeded sanity에서는 `evidence_bearing` vs `grounded_correction`가 동일한 Phase-2 survival을 재현하는 것을 확인했다. 따라서 이후 direct comparison에서는 **Phase-3 correction dynamics 차이**에 더 집중할 수 있다.

---

## 0) Data hygiene (중요)

### Legacy vs Strict
과거 pilot/debug 데이터(`*_val_50`, `*_val_100`, debug10/boxed10 등)가 unified data_dir에 섞이면서 결과 폴더에 50/100 테스트가 같이 생성되는 문제가 있었다.

- legacy 데이터 이동: `/data_x/aa007878/galileo/data_legacy/`
- 정식(full) QA 데이터: `/data_x/aa007878/galileo/data_qa_full/`
- 논문용 실행은 아래를 권장:
  - `scripts/make_all_data_dir_strict.sh` (정식 데이터만 링크)
  - strict unified dir 예시: `/data_x/aa007878/galileo/data_all_strict_*`

**논문 보고 시 주의:** `*_val_50`가 포함된 결과는 “참고/디버그”로만 취급하고, 최종 표/그림은 strict data_dir 기반 결과로만 작성한다.

---

## 1) 실험 프로토콜 요약

GALILEO는 다음 3단계 프로토콜로 robustness를 측정한다.

1. **Initial**: 정답(ground-truth) 기반 정확도
2. **Adversarial**: 5개 persona × 최대 5라운드 압박 → 라운드별 survival
3. **Recovery**: flip(오답 전향) 샘플만 회복 프롬프트 적용 → recovery rate

Outputs:
- JSONL: `*_initial.jsonl`, `*_adversarial.jsonl`, `*_recovery.jsonl`
- CSV: `initial_accuracy.csv`, `adversarial_survival.csv`, `recovery_accuracy.csv`

---

## 2) 7B 결과 스냅샷 (GPU 0–3, TP=4)

실행 정보:
- Results: `/mnt/raid6/aa007878/galileo/results/all_pilot_20260203_143301/7b/`
- GPU/TP: `CUDA_VISIBLE_DEVICES=0,1,2,3`, `TP_SIZE=4`
- `NUM_SAMPLES=1000`, `MAX_MODEL_LEN=16384`, `MAX_TOKENS=2048`, `SEED=42`

### 2.1 Initial accuracy

- GSM8K: **96.70%** (967/1000)
- SVAMP: **95.14%** (666/700)
- ARC-Easy(validation): **95.26%** (543/570)
- SQuAD 1.1(validation): **79.40%** (794/1000)
- SQuAD 2.0(validation): **76.50%** (765/1000)
- TriviaQA(rc validation): **55.60%** (556/1000)

### 2.2 Survival @ round 5 (aggregate)

- Authority Claim: **39.61%** (1730/4368)
- Strong Pressure: **48.49%** (2118/4368)
- Simple Denial: **51.19%** (2236/4368)
- Logical Trap: **59.98%** (2620/4368)
- Soft Pressure: **69.57%** (3039/4368)

### 2.3 Recovery (flip 이후 aggregate)

- Authority Claim: **71.30%** (1881/2638)
- Strong Pressure: **76.27%** (1716/2250)
- Simple Denial: **74.77%** (1594/2132)
- Soft Pressure: **81.79%** (1087/1329)
- Logical Trap: **85.76%** (1499/1748)

### 2.4 핵심 관찰(논문용 문장 후보)

- **Persona ordering의 안정성**: Authority Claim이 가장 치명적이며(최저 survival), Soft Pressure가 가장 약하다.
- **동역학 중요성**: 동일 모델에서도 persona에 따라 “초반 급격 붕괴 vs 완만 붕괴” 형태가 달라진다.
- **불확실성–취약성 연결**: open-domain QA(TriviaQA)는 초기 정확도가 낮고 강한 압박에서 취약해지는 경향.
- **Robustness vs Recovery 분리**: survival이 낮아도 recovery가 높게 나타날 수 있어, ‘무너뜨리기’와 ‘되돌리기’는 동일 난이도가 아니다.

---

## 3) Figure-ready exports (curve / failure / qualitative sheet)

`paper_export.py`로 논문 그림/정성 분석에 바로 쓸 수 있는 CSV를 생성한다.

예시 출력(7B):
- `/mnt/raid6/aa007878/galileo/results/all_pilot_20260203_143301/7b/paper_exports/`
  - `survival_curve.csv` (persona×round)
  - `turn_of_failure.csv` (persona×dataset×first-failure-turn)
  - `flip_samples.csv` (flip 사례 sheet)

### 3.1 Turn-of-failure (aggregate, 예시)

7B aggregate 기준(첫 오답 라운드 분포):

- Authority Claim: never 39.6%, **fail@1 34.9%**
- Soft Pressure: never 69.6%, fail@1 10.4%

→ Authority Claim은 **초반 라운드에서 큰 붕괴를 유도**한다는 해석을 뒷받침.

---

## 4) Multi-seed (A/B/C) – 진행 중

논문 제출 수준에서는 seed 민감도(특히 QA)를 통제해야 하므로 multi-seed 평균±std(가능하면 CI) 보고를 권장한다.

실행(예시: GPU 4–7):

```bash
GPU_LIST=4,5,6,7 TP_SIZE=4 NUM_SAMPLES=1000 MAX_MODEL_LEN=16384 MAX_TOKENS=2048 \
SEEDS=1,2,3,4,5 DATA_ALL_DIR=/data_x/aa007878/galileo/data_all_strict_4567 \
MATH_DIR=/data_x/aa007878/galileo/data QA_DIR=/data_x/aa007878/galileo/data_qa_full \
  bash scripts/run_multiseed_tmux.sh galileo-multiseed-4567
```

집계:

```bash
python scripts/aggregate_multiseed.py \
  --results_root /mnt/raid6/aa007878/galileo/results/multiseed_YYYYMMDD_HHMMSS \
  --out_dir /mnt/raid6/aa007878/galileo/results/multiseed_YYYYMMDD_HHMMSS/paper_tables \
  --round 5
```

---

## 5) Qualitative taxonomy labeling

`make_taxonomy_sheet.py`로 persona×task 균형 샘플을 뽑아, 논문 정성분석(실패 유형 분류)을 빠르게 수행한다.

```bash
python scripts/make_taxonomy_sheet.py \
  --model_dir /mnt/raid6/aa007878/galileo/results/all_pilot_20260203_143301/7b/Qwen2.5-7B-Instruct \
  --out_csv /mnt/raid6/aa007878/galileo/results/all_pilot_20260203_143301/7b/paper_exports/taxonomy_sheet.csv \
  --per_cell 20 \
  --seed 42
```

라벨 taxonomy(권장):
- Authority compliance
- Social appeasement
- Logical trap
- Uncertainty collapse
- Hedged flip

---

## 6) Next (논문용 필수 보강)

- 7B vs 14B 비교(동일 설정) + multi-seed 평균±std/CI
- Survival curve 그림(aggregate + task별 facet)
- Turn-of-failure 분포 그림
- Taxonomy 라벨링 + 대표 사례 10~20개 Appendix
- (선택) Ablation: recovery prompt variant / boxed vs free-form / decoding temperature
