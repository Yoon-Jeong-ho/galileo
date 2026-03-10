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

---

## 1) 현재까지 한 일 (요약)

### 1.1 실험(Experiments): “paper-ready(=auditable green)” 파이프라인 확립

**핵심 정책(SSOT):** EMNLP Main 실험은 원격에서 `ssh nlp8`, 레포 `/data_x/aa007878/galileo`, GPU는 **0–6 중에서 “진짜 idle + 타 유저 미사용 + CUDA alloc preflight OK”만** 사용, 모든 장기 작업은 `tmux`로 실행.

> 주의: heartbeat 배너에 `nlp16`/`CUDA_VISIBLE_DEVICES=4,5,6,7`가 등장하는 경우가 있는데, **stale**한 문구입니다. 실험 SSOT는 nlp8 기준으로만 운영합니다.

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

**현재 확보된 실험 증거(요지)**

- Qwen2.5-7B-Instruct 기반 **multi-seed** + Control vs Persona 비교가 "auditable green"으로 고정되어 있고,
  persona 유형별로 survival/TOF/recovery가 일관되게 변화함.
- **Cross-family generalization (Tier-1, seeds 1–2)**를 최소 비용으로 확장:
  - Llama 계열, Mistral 계열, Phi 계열 등에서 seeds 1–2 paper-ready 확보.
- **Decoding sensitivity (Tier-1, seeds 1–2)**: greedy temperature(0.0 vs 0.7) 변화에 대해 요약 아티팩트와 그림을 생성하여
  “결과가 디코딩 설정에 의해 완전히 뒤집히지 않음”을 확인.
- **Recovery-variant ablation (verify_then_answer; seeds 1–2)**: recovery 측정이 프롬프트/절차에 민감할 수 있음을
  통제된 방식으로 보여주는 ablation 아티팩트 확보.

> 최신 상태/정확한 run alias 목록은 `docs/paper/STATUS.md`의 NOW 섹션을 SSOT로 봅니다.

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

> 정의/용어 SSOT는 `docs/paper/FIGURE_CAPTIONS.md` 및 `docs/paper/STATUS.md`에도 반영되어 있습니다.

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

### 5.2 (원격) 필수 런북(SSOT)

- 실험 런북: `docs/paper/REMOTE_EXPERIMENTS_RUNBOOK.md`
- 상태판(가장 최신): `docs/paper/STATUS.md`

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

## 7) “지금까지” 진행 내역(타임라인을 찾는 법)

- 롤링 상태(중복 없는 현재 상태): `docs/paper/STATUS.md`
- Append-only 상세 타임라인: `docs/paper/HEARTBEAT_LOG.md`

> README에 모든 타임라인을 복붙하면 관리가 깨지므로,
> 이 README는 **구조/재현/타당성/SSOT 링크**에 집중하고,
> 이벤트 로그는 위 두 파일을 SSOT로 유지합니다.

---

## 8) 다음 액션(추천; reviewer-risk 기준)

1) **Abstract/Intro 문장별 proof-pointer(figure/table/artifact) 완성**
2) **cross-family 결과를 Results에 더 “정량문장”으로 반영** (seeds 1–2 범위에서)
3) **한 개 family 추가 or seed 확장 여부 결정** (CI가 흔들릴 때만 seed 확장)

---

## 9) 참고/부록

- Paper docs overview: `docs/paper/README.md`
- 정리된 선행연구/포지셔닝(KO): `docs/paper/LITERATURE_REVIEW_AND_POSITIONING_KO.md`
- 정량 결과 분석 노트(KO): `docs/paper/PAPER_RESULTS_ANALYSIS_KO.md`
- 정성 예시(KO): `docs/paper/PAPER_RESULTS_QUAL_EXAMPLES_KO.md`
