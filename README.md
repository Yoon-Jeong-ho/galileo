# GALILEO (EMNLP Main) — 진행 현황/재현 가이드 (KO)

> TL;DR: **정답(ground-truth)이 있는 과제**에서 LLM이 multi-turn 상호작용 중 **페르소나 기반 압박(persona pressure)**에 의해
> (i) 정답을 얼마나 오래 **유지(survival)**하는지,
> (ii) 최초로 무너지는 시점인 **TOF (turn-of-failure)**가 언제인지,
> (iii) 한 번 flip(오답 전향) 이후 **회복(recovery; flip 조건부)**이 가능한지를 **동일 프로토콜**로 계량합니다.
> 또한 동일 라운드/디코딩에서 비공격적 반복질문인 **Neutral Re-asking Control(드리프트 베이스라인)**을 포함해, 단순 드리프트와 압박 효과를 분리합니다.

---

## 0) 이 README의 목적

이 문서는 "지금까지 무엇을 했는가"를 **실험/코드/논문조사/논문작성** 관점에서 하나로 묶어 설명하고,
**어떤 산출물이 논문에 인용 가능한지(타당성/검증 상태)**와 **재현 방법**을 단계별로 정리합니다.

- 이전 README는 보존했습니다: `README_prev_20260217.md` (그리고 더 오래된 요약: `Prev_README.md`).

### 0.1 현재 로컬 코드 상태 (2026-03-10 확인)

아래는 **이 체크아웃에서 직접 확인한 사실**입니다.

- 현재 워킹트리에는 `data/`, `results/`, `results_paper/` 디렉터리가 **없습니다**. 따라서 로컬에서 `run_experiment.py`를 기본값으로 바로 실행하면 데이터 경로를 명시적으로 넘기거나, 먼저 데이터 디렉터리를 준비해야 합니다.
- 코드 기본 경로는 이제 **하드코딩된 `/data_x/.../galileo`** 대신 **현재 레포 루트 기준(`./data`, `./results`)** 또는 환경변수(`GALILEO_DATA_DIR`, `GALILEO_RESULTS_DIR`)를 사용하도록 정비했습니다.
- 기본 GPU 가시성도 이제 **단일 GPU 안전 기본값**을 따르며, 실제 실험에서는 반드시 `CUDA_VISIBLE_DEVICES`를 명시적으로 설정하는 것을 권장합니다.
- `scripts/remote_run/nlp8_smoke.sh`, `scripts/run_multiseed_tmux.sh`, `scripts/run_multiseed_families_tmux.sh`도 **스크립트 위치 기준으로 repo root를 추론**하도록 정비했습니다. 기존 원격 SSOT 경로를 그대로 쓰고 싶으면 `REPO_DIR=...`로 명시하면 됩니다.
- 비-GPU smoke 성격 검증은 `python -m unittest tests/test_non_gpu_core.py`로 수행할 수 있습니다.
- Recovery 프롬프트에는 기존 baseline 계열 외에 **`grounded_correction` / `evidence_bearing`** 변형을 추가했습니다. 데이터에 `correction_evidence`/`evidence`/`supporting_facts`/`explanation` 등이 있으면 그 근거를 사용하고, 없으면 verified answer를 명시하는 answer-bearing correction으로 동작합니다.
- **GPU 5 가용 메모리가 부분 점유된 경우** `--gpu_memory_utilization 0.6`으로도 single-GPU sanity를 통과시킬 수 있음을 2026-03-10에 확인했습니다.
- **중요 수정(2026-03-10 late):** generation seed가 Phase 1/2/3에 명시적으로 전달되도록 수정했습니다. 이 수정 전에는 `evidence_bearing` vs `grounded_correction` 비교에서 recovery variant만 바뀌어도 Phase 2 dynamics가 run-to-run stochasticity로 달라질 수 있었습니다. 현재는 seeded sanity에서 phase-2 결과가 일치하는 것을 확인했습니다.

---

## 1) 현재까지 한 일 (요약)

### 1.1 실험(Experiments): 현재 기준선(2026-03-10 검증)

현재 README가 기준으로 삼는 **직접 검증된 실험 기준선**은 아래입니다.

- **Synthetic smoke run**
  - 결과: `/data_x/aa007878/projects/galileo/tmp/results/smoke_gpu5_20260310_184715/`
  - 검증: `paper_exports/` 생성 + validator `[OK]`
- **Real-data small pilot (math)**
  - 결과: `/data_x/aa007878/projects/galileo/tmp/results/pilot_gpu5_real_20260310_185233/`
  - 데이터: GSM8K + SVAMP, 각 5 샘플
  - arms: `control_reask`, `authority_claim`, `evidence_bearing` recovery
  - 검증: `paper_exports/` 생성 + validator `[OK]`
- **Real-data pilot (math)**
  - 결과: `/data_x/aa007878/projects/galileo/tmp/results/pilot50_gpu5_20260310_185825/`
  - 데이터: GSM8K + SVAMP, 각 50 샘플
  - arms: `control_reask`, `authority_claim`, `evidence_bearing` recovery
  - 검증: `paper_exports/` 생성 + validator `[OK]`
- **Non-math main (MCQA)**
  - 결과: `/data_x/aa007878/projects/galileo/tmp/results/main_arc_gpu6_20260310_191906/`
  - 데이터: ARC-Easy 50 샘플
  - arms: `control_reask`, `authority_claim`, `evidence_bearing` recovery
  - 검증: `paper_exports/` 생성 + validator `[OK]`
- **논문 후보 승격(results_paper)**
  - math: `/data_x/aa007878/projects/galileo/results_paper/qwen7b_math_control_authority_evidence_gsm8k_svamp_gpu5_20260310/`
  - non-math: `/data_x/aa007878/projects/galileo/results_paper/qwen7b_nonmath_control_authority_evidence_arc_gpu6_20260310/`
  - grounded math: `/data_x/aa007878/projects/galileo/results_paper/qwen7b_gsm8k_control_authority_grounded_gpu5_20260310/`
  - grounded non-math: `/data_x/aa007878/projects/galileo/results_paper/qwen7b_arc_control_authority_grounded_gpu5_20260310/`
  - root validator: `[OK] runner_metadata parity`

**Paper-ready(인용 가능) 판정 기준**

- 각 run 폴더가 다음을 포함:
  - `paper_exports/survival_curve.csv`
  - `paper_exports/turn_of_failure.csv`
  - `paper_exports/flip_samples.csv`
  - `paper_exports/metadata.json`
  - `paper_exports/runner_metadata.json`
- 검증 스크립트 통과:
  - `python3 scripts/validate_paper_exports.py --results_root <RUN_DIR>`
  - (paper SSOT root) `python3 scripts/validate_paper_exports.py --results_root results_paper --check_runner_parity`

**2026-03-10 기준 확인된 패턴(직접 실행 결과)**

- GSM8K 50샘플 pilot에서 `Authority Claim` survival@5는 **58.14%**, `Control Re-asking` survival@5는 **88.37%**
- SVAMP 50샘플 pilot에서 `Authority Claim` survival@5는 **75.00%**, `Control Re-asking` survival@5는 **93.75%**
- ARC-Easy 50샘플 main run에서 `Authority Claim` survival@5는 **24.49%**, `Control Re-asking` survival@5는 **93.88%**
- 위 세 run 모두에서 `evidence_bearing` recovery는 현재까지 **100%** recovery@flip을 보였음
- grounded main comparison (seeded-evidence와 직접 비교용):
  - GSM8K grounded main: authority survival@5 **63.41%**, control survival@5 **92.68%**, authority recovery@flip **93.33%**
  - ARC-Easy grounded main: authority survival@5 **36.73%**, control survival@5 **91.84%**, authority recovery@flip **100.00%**

> 해석 주의: 위 수치는 아직 **2026-03-10 seed-1 기준선**이며, 논문 headline claim은 더 큰 샘플/다중 seed 확인 후 강화해야 합니다.

### 1.2 코드(Code): “실험→export→검증→paper artifact/figure” 연결

- run 결과에서 paper에 바로 쓰는 최소 산출물(`paper_exports/*`)을 표준화.
- **Tracked artifacts**(논문 주장에 직접 연결되는 CSV)를 `docs/paper/artifacts/`에 고정.
- 위 artifacts로부터 **재생성 가능한 벡터 그림(SVG)**를 `docs/paper/figures/`에 생성.
- SVG→PDF 변환을 **sudo 없이** 가능한 경로(AppImage Inkscape)까지 제공하여 LaTeX 빌드 리스크 감소.
- 익명화 번들(packaging) 스크립트로 외부 공유/제출 시 인프라 문자열 누출을 fail-fast로 차단.

### 1.3 논문 조사(Research): 포지셔닝 강화

- multi-turn sycophancy/flip dynamics 계열(SYCON/Truth Decay 등)과의 **차별점**을 "ground-truth + survival/TOF + recovery + neutral drift control"로 정리.
- evidence-based belief revision(ReviseQA)와의 대비를 통해
  본 작업이 "새 증거에 의한 합리적 수정"이 아니라 "압박/드리프트"를 분리해 측정하는 설계임을 명확히 함.

관련 노트 SSOT:
- `docs/paper/LITERATURE_REVIEW_AND_POSITIONING_KO.md`
- `docs/paper/related_work/INDEX.md`

### 1.4 논문 작성(Writing): 주장↔근거 매핑과 재현성 문서화

- EN 드래프트를 기준으로, 주요 claim마다 **그림/표 라벨 + artifact 경로**로 proof-pointer를 고정.
- 캡션 SSOT를 별도 파일로 분리하여(캡션 드리프트 방지) provenance(어느 artifact에서 나온 그림인지)를 명시.

핵심 파일:
- `docs/paper/PAPER_DRAFT_EN.md` (메인)
- `docs/paper/PAPER_DRAFT_KO.md` (한글 초안/메모)
- `docs/paper/FIGURE_CAPTIONS.md` (캡션+provenance)
- `docs/paper/CLAIM_EVIDENCE_MAP.md` (claim→evidence)

---

## 2) 레포 구조(어디에 뭐가 있나)

### 2.1 실험 코드(핵심 엔트리)

- `run_experiment.py` : 실험 파이프라인 엔트리(Initial → Persona/Control multi-turn → Recovery → export)
- `personas.py` : 5개 adversarial persona 프롬프트(압박 유형 정의)
- `tasks.py`, `data_loader.py` : 태스크/데이터 로딩 및 포맷
- `evaluation.py`, `inference.py` : 추론 및 채점/평가 로직
- `config.py` : 기본 설정(모델 목록, 토큰/라운드/빔 등)

### 2.2 Paper 파이프라인(논문 산출물 SSOT)

- Paper 문서 루트: `docs/paper/`
- 논문 주장에 직접 쓰는 CSV(추적): `docs/paper/artifacts/`
- 그림(SVG; 재생성 소스): `docs/paper/figures/`
- PDF 그림(LaTeX용; 필요 시): `paper_figures/pdf/`

---

## 3) 프로토콜 정의(무엇을 측정하는가) — 핵심 개념

### 3.1 3-Phase 평가

1) **Initial**: 모델이 문제를 “혼자 풀 때” 정답을 맞히는가?
2) **Multi-turn pressure**:
   - (a) **Persona pressure**: 공격자 페르소나가 라운드별로 압박/반박
   - (b) **Neutral Re-asking Control**: 공격 없이 반복질문(새 증거 없음) → 드리프트 베이스라인
3) **Recovery (flip 조건부)**: persona pressure 과정에서 **오답으로 flip된 케이스**에 대해 회복 프롬프트를 주고
   정답으로 돌아오는지 측정

### 3.2 지표

- **Survival(p, r)**: persona p 하에서 round r까지 “계속 정답 유지(누적)”한 비율
- **TOF (turn-of-failure)**: 최초로 오답이 등장한 라운드의 분포(없으면 never)
- **Recovery@flip**: flip된 케이스에 한정하여 정답으로 복귀한 비율
- **Effect(Δ)**: Persona vs Control 간의 ΔSurvival@5, ΔFail@1, ΔRecovery@flip 등

> 정의/용어 SSOT는 `docs/paper/FIGURE_CAPTIONS.md`를 우선 기준으로 봅니다.

---

## 4) 타당성(Validity) / 왜 믿을만한가

### 4.1 내부 타당성: “압박 효과”와 “드리프트” 분리

- 같은 라운드/디코딩/템플릿 구조에서 **Control을 함께 측정**하여,
  multi-turn 자체가 유발하는 일반적인 drift(피로/일관성 붕괴 등)와
  persona 텍스트가 유발하는 압박 효과를 분리하도록 설계.

### 4.2 재현성/감사 가능성: auditable green 정책

- 각 run마다 `paper_exports/*` + 메타데이터를 남기고 validator로 강제.
- 논문에 인용하는 수치는 **git에 트래킹되는 artifacts(CSV)**로 고정.
- 그림은 artifacts에서 stdlib 스크립트로 재생성 가능(SVG SSOT).

### 4.3 외적 타당성: cross-family 최소 확장

- EMNLP 리스크 감소 목적의 Tier‑1 정책으로, 추가 모델 family를 seeds 1–2로 확장해
  “특정 모델 특이 결과” 가능성을 낮춤.

### 4.4 현재 한계(명시적으로 남은 리스크)

- 데이터/태스크가 정답형 중심이며, 도메인 확장(정답형 QA/사실검증/논리) 시 추가 검증 필요.
- decoding/프롬프트 선택이 측정값에 영향을 줄 수 있어, 본 레포는 이를 **ablation으로 드러내고 통제**하는 방향.
- 모델 정책/시스템프롬프트에 의존할 수 있어, 최종 논문에서는 프롬프트/설정 공개 수준을 명확히 해야 함.

---

## 5) 재현(How-to) — 가장 현실적인 실행 경로

> 실험은 보통 원격(nlp8)에서 수행하고, 이 레포(로컬)는 논문/아티팩트/그림/익명화 번들 생성에 주로 사용합니다.

### 5.1 (로컬) 환경 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> torch/vLLM은 CUDA/드라이버에 강하게 의존하므로 환경에 맞게 별도 설치를 권장합니다.

현재 실무 권장(이 레포의 최근 작업 기준):

```bash
source /data_x/aa007878/miniconda3/etc/profile.d/conda.sh
conda activate galileo
```

### 5.1.1 로컬 코드 하드닝 상태 (2026-03-10 확인)

검증된 사실:

- `config.py` 기본 경로는 이제 **레포 기준 상대 경로**(`data/`, `results/`)를 사용하며,
  필요 시 `GALILEO_DATA_DIR`, `GALILEO_RESULTS_DIR`로 override 가능
- `run_experiment.py` / `config.py`는 더 이상 `CUDA_VISIBLE_DEVICES`를 **암묵적으로 강제하지 않음**
- `scripts/run_multiseed_tmux.sh`, `scripts/run_multiseed_families_tmux.sh`, `scripts/remote_run/nlp8_smoke.sh`
  의 안전 기본값은 현재 **GPU 7 / TP=1** 기준으로 맞춰 둠 (필요 시 env override)
- 데이터셋 row에 `correction_evidence` / `supporting_evidence` / `supporting_facts` / `explanation` / `rationale`
  중 하나가 있으면 evidence-bearing correction arm에서 사용할 수 있도록 로더가 보존함
- recovery variant로 `grounded_correction` / `evidence_bearing`를 사용할 수 있음
- 현재 로컬 checkout에는 `data/`, `results/`, `results_paper/`가 없을 수 있으므로, 실험 재개 시 경로를 명시하거나 remote SSOT를 사용해야 함
- **GPU 5 smoke run (2026-03-10)**: `Qwen/Qwen2.5-7B-Instruct`, `tensor_parallel_size=1`,
  `tmp/smoke_data/{smoke_math,smoke_mcqa}.jsonl`, personas=`control_reask,authority_claim`,
  recovery=`evidence_bearing` 경로로 end-to-end 실행 성공
  - raw results: `tmp/results/smoke_gpu5_20260310_184715/`
  - paper exports + validator: `tmp/results/smoke_gpu5_20260310_184715/paper_exports/` / `[OK] runner_metadata parity`
- **GPU 5 real-data pilot (2026-03-10, small-n)**: `Qwen/Qwen2.5-7B-Instruct`, `tensor_parallel_size=1`,
  data=`gsm8k + svamp`, `num_samples=5` each, personas=`control_reask,authority_claim`,
  recovery=`evidence_bearing` 경로로 실행 성공
  - raw results: `tmp/results/pilot_gpu5_real_20260310_185233/`
  - paper exports + validator: `tmp/results/pilot_gpu5_real_20260310_185233/paper_exports/` / `[OK] runner_metadata parity`
  - quick read (pilot only, **not headline**):
    - GSM8K initial 5/5, Authority Claim survival@5 20%, Control survival@5 100%, Recovery@flip 100%
    - SVAMP initial 4/5, Authority Claim survival@5 50%, Control survival@5 100%, Recovery@flip 100%

비-GPU 코드 검증(로컬에서 즉시 가능):

```bash
python -m py_compile \
  config.py data_loader.py evaluation.py inference.py personas.py run_experiment.py tasks.py \
  scripts/paper_export.py scripts/write_runner_metadata.py scripts/validate_paper_exports.py

python -m unittest discover -s tests -v
```

단일 GPU 실험을 재개할 때는 항상 명시적으로:

```bash
CUDA_VISIBLE_DEVICES=7 python run_experiment.py ...
```

또는 데이터/결과 경로를 함께 명시:

```bash
CUDA_VISIBLE_DEVICES=7 \
GALILEO_DATA_DIR=/path/to/data \
GALILEO_RESULTS_DIR=/path/to/results \
python run_experiment.py ...
```

evidence-bearing correction arm을 켜려면 예를 들어:

```bash
CUDA_VISIBLE_DEVICES=7 \
python run_experiment.py \
  --data_file /path/to/smoke.jsonl \
  --results_dir /path/to/results \
  --tensor_parallel_size 1 \
  --recovery_variant evidence_bearing
```

> 주의: 위 variant는 **코드 경로만 구현/검증**된 상태이며, headline 실험 결과는 아직 다시 생성하지 않았습니다.

### 5.2 (원격) 참고 런북(legacy support)

- 참고용 런북: `docs/paper/REMOTE_EXPERIMENTS_RUNBOOK.md`
- 단, **현재 source of truth는 README + 실제 코드 + 2026-03-10 결과물**입니다.
- `STATUS.md`, `HEARTBEAT_LOG.md` 등 과거 운영 문서는 현재 워크플로의 SSOT가 아닙니다.

원격에서 매번 최소 확인:

```bash
ssh nlp8
cd /data_x/aa007878/galileo

tmux ls
nvidia-smi -i 4,5,6
# 가장 최근 run.log tail
```

### 5.3 paper_exports 검증

```bash
python3 scripts/validate_paper_exports.py --results_root results_paper --check_runner_parity
```

### 5.3.1 Table 1 자동 재생성(권장; 표가 빈약해 보이는 문제 해결)

Table 1(메인 테이블)에 들어가는 Survival/Fail@1/Recovery@flip 값은 **results_paper의 paper_exports에서 자동 추출**되어 `docs/paper/artifacts/`에 고정됩니다.

```bash
# (nlp8) results_paper/*/paper_exports로부터 Table-1 셀을 추출해 artifact CSV 생성
python3 scripts/make_table1_from_results_paper_exports.py \
  --results_paper results_paper \
  --out docs/paper/artifacts/table1_from_results_paper_exports_$(date +%Y%m%d).csv

# (로컬/원격) artifact로부터 LaTeX Table-1 rows 생성
python3 scripts/gen_latex_table1_from_artifacts.py \
  --out docs/paper/latex_paper_emnlp2023/generated/table1_rows.tex
```

> Recovery@flip은 `paper_exports/recovery_accuracy.csv`가 export된 run에서만 채워집니다. (구버전 paper_exports에는 없을 수 있음)

### 5.4 (로컬) artifacts → SVG figures 재생성

- 그림 생성 스크립트들은 `scripts/`에 있으며, 결과는 `docs/paper/figures/`로 저장됩니다.
- SVG→PDF 변환(LaTeX용):

```bash
bash scripts/get_inkscape_appimage.sh
./scripts/convert_figures_svg_to_pdf.sh
bash scripts/check_pdf_figures.sh
```

### 5.5 익명화 번들(외부 공유/제출용)

```bash
./scripts/package_anonymized_bundle.sh
./scripts/archive_anonymized_bundle.sh tmp/anonymized_bundle tmp/galileo_anonymized_bundle
```

---

## 6) 논문 작성/관리 — 무엇을 어디서 업데이트해야 하나

- Paper index: `docs/paper/README.md`
- EN draft(메인): `docs/paper/PAPER_DRAFT_EN.md`
- Captions/provenance SSOT: `docs/paper/FIGURE_CAPTIONS.md`
- Claim→Evidence 매핑: `docs/paper/CLAIM_EVIDENCE_MAP.md`
- Submission checklist: `docs/paper/EMNLP_MAIN_SUBMISSION_CHECKLIST.md`

---

## 7) 현재 활성 문서 우선순위

- 1순위: `/data_x/aa007878/projects/galileo/README.md`
- 2순위: 현재 실제 코드 (`run_experiment.py`, `config.py`, `personas.py`, `scripts/*`)
- 3순위: `/data_x/aa007878/projects/galileo/tmp/results/*20260310*/`
- 4순위: `/data_x/aa007878/projects/galileo/docs/paper/PAPER_DRAFT_EN.md`
- 5순위: `/data_x/aa007878/projects/galileo/docs/paper/CLAIM_EVIDENCE_MAP.md`

> `STATUS.md`, `HEARTBEAT_LOG.md` 등 과거 운영 문서는 **기본 워크플로에서 유지보수하지 않습니다.**

---

## 8) 다음 액션(추천; reviewer-risk 기준)

1) **math + non-math direct comparison evidence set 완성**
2) **control vs authority vs evidence-bearing 결과를 표/문장으로 고정**
3) **다중 seed(≥3) 확장 여부 결정**

---

## 9) 참고/부록

- Paper docs overview: `docs/paper/README.md`
- 정리된 선행연구/포지셔닝(KO): `docs/paper/LITERATURE_REVIEW_AND_POSITIONING_KO.md`
- 정량 결과 분석 노트(KO): `docs/paper/PAPER_RESULTS_ANALYSIS_KO.md`
- 정성 예시(KO): `docs/paper/PAPER_RESULTS_QUAL_EXAMPLES_KO.md`
