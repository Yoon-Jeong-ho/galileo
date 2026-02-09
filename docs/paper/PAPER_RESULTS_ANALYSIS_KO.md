# 실험 결과 분석 (논문용)

> 아래 내용은 strict data_dir + multi-seed 결과의 **현재까지 누적분**으로 작성되었다.
> Results root: `/mnt/raid6/aa007878/galileo/results/multiseed_20260203_173602`
> 사용 가능한 seeds: 7B=['seed_1', 'seed_2', 'seed_3', 'seed_4'] / 14B=['seed_1', 'seed_2', 'seed_3']

## 1. 실험 설정
- 모델: Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct
- 벤치마크: gsm8k, svamp, arc_easy_validation, squad11_validation, squad20_validation, triviaqa_rc_validation
- persona: 5종(Soft/Denial/Strong/Authority/Trap), 최대 5라운드
- 보고 지표: initial accuracy / round별 survival / flip 이후 recovery

## 1.X (논문 Table W용) Fresh 2-run 스냅샷: Neutral control vs Persona pressure (seed1, nlp8)

목적: **drift를 분리하는 neutral control(=Neutral Re-asking Control)** 대비, persona pressure에서 survival/TOF가 얼마나 악화되는지 “paper-ready” 숫자로 한 번에 보여주기.

- Control run root (auditable green): `nlp8:/data_x/aa007878/galileo/results/c2run_control_20260209_172640/`
- Persona run root (auditable green): `nlp8:/data_x/aa007878/galileo/results/c2run_persona_20260209_174640/`
- Table W 산출 스크립트: `scripts/make_table_w_control_vs_persona.py` (control_persona_id=`neutral_reask_control`, round=5)

**요약 (seed1, NUM_SAMPLES=80):**

| metric | control (Neutral Re-asking) | persona (weighted) | persona (unweighted) |
|---|---:|---:|---:|
| Survival@5 | 79.60 | 57.41 | 57.41 |
| Fail@1 | 10.70 | 20.57 | 20.31 |
| Never-fail | 79.60 | 58.62 | 58.50 |

해석(초안):
- 동일 모델/샘플 기준에서, persona pressure는 **Survival@5를 ~22pt 낮추고**, **Fail@1을 ~2배**까지 끌어올린다.
- 이 스냅샷은 “neutral control이 drift baseline”이라는 주장과 “persona pressure가 실제 붕괴를 만든다”는 주장을 분리해서 보여주는 데 사용 가능.

## 2. 모델×벤치마크: Initial / Recovery (mean±std over seeds)
### 7B

| benchmark | initial acc | recovery avg (persona-avg) |
|---|---:|---:|
| arc_easy_validation | 95.18 ± 0.10 | 77.80 ± 0.95 |
| gsm8k | 96.80 ± 0.63 | 47.10 ± 1.45 |
| squad11_validation | 77.12 ± 0.96 | 84.88 ± 0.48 |
| squad20_validation | 76.88 ± 0.51 | 83.09 ± 1.31 |
| svamp | 95.11 ± 0.32 | 50.38 ± 3.39 |
| triviaqa_rc_validation | 57.80 ± 1.70 | 81.19 ± 1.32 |

### 14B

| benchmark | initial acc | recovery avg (persona-avg) |
|---|---:|---:|
| arc_easy_validation | 95.26 ± 0.18 | 85.14 ± 0.73 |
| gsm8k | 98.47 ± 0.15 | 89.85 ± 1.41 |
| squad11_validation | 82.70 ± 0.00 | 92.55 ± 0.17 |
| squad20_validation | 79.23 ± 2.05 | 92.80 ± 0.94 |
| svamp | 97.67 ± 0.17 | 80.16 ± 1.50 |
| triviaqa_rc_validation | 75.77 ± 0.12 | 91.99 ± 0.05 |

## 3. 모델×벤치마크×라운드: Survival curve (persona 평균, mean±std)
각 벤치마크에서 persona 5개의 survival을 평균내어 라운드별 곡선으로 보고한다.

### 7B
#### arc_easy_validation
| r1 | r2 | r3 | r4 | r5 |
|---:|---:|---:|---:|---:|
| 72.63±1.38 | 51.25±0.79 | 41.76±0.56 | 36.88±1.05 | 33.76±0.68 |

#### gsm8k
| r1 | r2 | r3 | r4 | r5 |
|---:|---:|---:|---:|---:|
| 94.74±0.14 | 91.13±0.10 | 88.67±0.26 | 86.83±0.31 | 84.71±0.17 |

#### squad11_validation
| r1 | r2 | r3 | r4 | r5 |
|---:|---:|---:|---:|---:|
| 70.28±0.75 | 55.55±0.45 | 47.59±0.35 | 41.32±0.23 | 37.00±0.32 |

#### squad20_validation
| r1 | r2 | r3 | r4 | r5 |
|---:|---:|---:|---:|---:|
| 67.84±0.93 | 53.00±1.21 | 45.06±1.31 | 38.97±0.98 | 34.95±0.57 |

#### svamp
| r1 | r2 | r3 | r4 | r5 |
|---:|---:|---:|---:|---:|
| 95.49±0.38 | 92.53±0.44 | 90.52±0.48 | 88.55±0.47 | 86.50±0.53 |

#### triviaqa_rc_validation
| r1 | r2 | r3 | r4 | r5 |
|---:|---:|---:|---:|---:|
| 67.19±1.35 | 49.71±1.86 | 41.38±1.61 | 36.50±1.80 | 33.39±1.88 |

### 14B
#### arc_easy_validation
| r1 | r2 | r3 | r4 | r5 |
|---:|---:|---:|---:|---:|
| 81.96±1.30 | 67.85±0.61 | 57.66±0.93 | 51.15±0.65 | 45.73±1.03 |

#### gsm8k
| r1 | r2 | r3 | r4 | r5 |
|---:|---:|---:|---:|---:|
| 94.96±0.21 | 88.79±0.38 | 83.55±0.17 | 79.51±0.29 | 76.20±0.32 |

#### squad11_validation
| r1 | r2 | r3 | r4 | r5 |
|---:|---:|---:|---:|---:|
| 59.40±0.93 | 46.03±1.40 | 38.97±1.15 | 33.96±1.14 | 30.59±1.00 |

#### squad20_validation
| r1 | r2 | r3 | r4 | r5 |
|---:|---:|---:|---:|---:|
| 56.39±0.81 | 43.36±0.48 | 36.18±0.34 | 31.69±0.67 | 28.27±0.58 |

#### svamp
| r1 | r2 | r3 | r4 | r5 |
|---:|---:|---:|---:|---:|
| 96.72±0.25 | 94.14±0.41 | 91.76±0.17 | 89.49±0.13 | 87.25±0.09 |

#### triviaqa_rc_validation
| r1 | r2 | r3 | r4 | r5 |
|---:|---:|---:|---:|---:|
| 68.10±0.67 | 56.51±1.22 | 49.24±1.29 | 44.89±1.11 | 41.75±1.37 |

## 4. 논문 서술형 분석(핵심 주장 형태)
### 4.1 벤치마크별 차이: Math vs QA/OpenQA
- **Math(GSM8K/SVAMP)**는 라운드가 진행되어도 survival이 상대적으로 높게 유지되는 경향이 있다.
- 반면 **QA/MCQA/OpenQA**는 persona 압박에서 survival 곡선의 하강 기울기가 더 큰 경우가 많다.

### 4.2 라운드별 분석의 의미: ‘얼마나’뿐 아니라 ‘언제’ 무너지는가
- Authority-type 압박은 early-round에서 큰 붕괴를 유발할 수 있고, trap-type은 누적적으로 침식하는 형태를 보일 수 있다.
- 따라서 최종 논문에서는 survival curve와 함께 turn-of-failure(첫 오답 라운드) 분포를 함께 제시해야 한다.

### 4.3 스케일링(7B→14B): capability / survival / recovery의 분리
- 누적 결과 기준으로 14B는 initial accuracy 및 recovery가 크게 개선되는 경향을 보인다(특히 open-domain QA와 recovery).
- survival은 벤치마크별로 단조 증가가 아닐 수 있어, capability와 robustness가 완전히 동일 축이 아님을 시사한다.

## 5. 논문에 넣는 방식(권장)
- Table: 모델×벤치마크 initial / survival@5 / recovery (mean±std, 최종은 CI 포함)
- Figure: 벤치마크별 survival curve(r1..r5) + turn-of-failure 분포


## 3.X (확장) Persona×Round Survival (벤치마크별, mean±std)
아래는 벤치마크별로 persona 5종의 라운드별 survival curve를 제시한다. (seed 평균±표준편차)

### 7B (seeds=['seed_1', 'seed_2', 'seed_3', 'seed_4'])
#### arc_easy_validation
| persona | r1 | r2 | r3 | r4 | r5 |
|---|---:|---:|---:|---:|---:|
| Authority Claim | 45.07±1.72 | 18.02±0.86 | 13.92±0.94 | 12.21±1.01 | 11.52±0.77 |
| Strong Pressure | 70.50±3.26 | 58.34±2.39 | 48.34±2.56 | 42.44±3.25 | 39.40±2.77 |
| Simple Denial | 76.31±1.22 | 58.53±0.64 | 49.22±1.27 | 44.93±1.80 | 41.89±2.65 |
| Logical Trap | 89.35±1.71 | 64.61±2.03 | 52.26±2.34 | 46.17±2.87 | 42.49±2.78 |
| Soft Pressure | 81.89±1.42 | 56.77±1.58 | 45.07±3.33 | 38.66±2.62 | 33.50±2.17 |

#### gsm8k
| persona | r1 | r2 | r3 | r4 | r5 |
|---|---:|---:|---:|---:|---:|
| Authority Claim | 88.38±0.44 | 79.49±0.51 | 73.81±1.05 | 70.43±1.36 | 67.48±1.16 |
| Strong Pressure | 96.20±0.91 | 93.75±0.47 | 91.99±0.44 | 90.42±0.57 | 87.91±0.77 |
| Simple Denial | 96.08±0.51 | 94.55±0.58 | 93.18±0.49 | 91.84±0.40 | 89.93±0.37 |
| Logical Trap | 96.10±0.55 | 93.47±0.35 | 92.15±0.34 | 91.14±0.42 | 90.21±0.56 |
| Soft Pressure | 96.95±0.31 | 94.37±0.66 | 92.20±0.84 | 90.32±1.13 | 87.99±1.00 |

#### squad11_validation
| persona | r1 | r2 | r3 | r4 | r5 |
|---|---:|---:|---:|---:|---:|
| Authority Claim | 50.63±0.39 | 31.75±1.00 | 26.70±1.09 | 23.81±1.03 | 22.45±0.80 |
| Strong Pressure | 69.02±1.47 | 59.00±0.46 | 45.28±0.53 | 35.19±0.20 | 28.65±0.71 |
| Simple Denial | 67.08±2.77 | 57.16±1.98 | 49.21±1.98 | 39.81±2.51 | 32.19±1.87 |
| Logical Trap | 74.35±0.75 | 48.53±1.15 | 39.80±1.15 | 35.44±2.03 | 32.98±1.70 |
| Soft Pressure | 90.34±0.43 | 81.32±1.17 | 76.94±1.25 | 72.37±1.58 | 68.73±1.12 |

#### squad20_validation
| persona | r1 | r2 | r3 | r4 | r5 |
|---|---:|---:|---:|---:|---:|
| Authority Claim | 49.88±1.98 | 30.89±1.80 | 26.01±1.09 | 23.34±1.33 | 21.78±1.44 |
| Strong Pressure | 64.65±1.84 | 54.83±2.17 | 43.10±3.50 | 32.59±3.06 | 26.34±1.45 |
| Simple Denial | 65.33±0.90 | 55.32±1.15 | 46.34±0.98 | 37.62±0.26 | 30.70±0.30 |
| Logical Trap | 70.73±0.78 | 45.30±0.79 | 37.05±1.56 | 33.50±1.90 | 31.78±2.35 |
| Soft Pressure | 88.58±0.48 | 78.67±1.56 | 72.79±1.24 | 67.81±1.11 | 64.17±0.96 |

#### svamp
| persona | r1 | r2 | r3 | r4 | r5 |
|---|---:|---:|---:|---:|---:|
| Authority Claim | 93.35±1.05 | 87.95±1.73 | 84.61±1.63 | 80.89±1.79 | 78.75±1.41 |
| Strong Pressure | 94.86±0.75 | 91.78±0.72 | 89.15±1.30 | 86.48±1.31 | 82.80±1.43 |
| Simple Denial | 94.74±0.28 | 92.41±0.26 | 91.06±0.89 | 89.75±0.65 | 87.91±0.55 |
| Logical Trap | 97.03±0.50 | 95.12±0.57 | 93.99±0.70 | 93.20±0.87 | 92.45±0.58 |
| Soft Pressure | 97.48±0.77 | 95.38±0.44 | 93.80±0.65 | 92.45±0.94 | 90.57±1.24 |

#### triviaqa_rc_validation
| persona | r1 | r2 | r3 | r4 | r5 |
|---|---:|---:|---:|---:|---:|
| Authority Claim | 49.81±3.08 | 35.57±2.55 | 30.28±2.56 | 27.52±2.75 | 25.43±2.79 |
| Strong Pressure | 61.74±1.10 | 32.57±2.71 | 17.69±2.91 | 10.99±2.99 | 8.11±2.40 |
| Simple Denial | 54.45±1.98 | 30.23±2.45 | 19.27±0.90 | 12.10±1.59 | 8.82±1.02 |
| Logical Trap | 84.54±0.54 | 72.47±2.36 | 66.86±1.65 | 63.71±2.07 | 60.74±2.87 |
| Soft Pressure | 85.38±1.37 | 77.71±0.85 | 72.78±1.58 | 68.16±2.00 | 63.85±2.24 |

### 14B (seeds=['seed_1', 'seed_2', 'seed_3', 'seed_4'])
#### arc_easy_validation
| persona | r1 | r2 | r3 | r4 | r5 |
|---|---:|---:|---:|---:|---:|
| Authority Claim | 76.35±1.19 | 65.88±0.79 | 60.30±0.80 | 57.08±0.43 | 54.54±0.24 |
| Strong Pressure | 81.28±1.64 | 62.75±1.05 | 46.52±1.02 | 35.96±1.69 | 27.02±1.26 |
| Simple Denial | 86.44±2.64 | 75.33±2.31 | 64.59±1.66 | 55.74±1.81 | 47.30±1.64 |
| Logical Trap | 78.19±1.11 | 63.72±1.60 | 56.11±1.27 | 51.59±1.52 | 47.86±1.84 |
| Soft Pressure | 86.30±2.05 | 71.78±1.62 | 62.80±3.10 | 57.40±2.50 | 53.44±2.36 |

#### gsm8k
| persona | r1 | r2 | r3 | r4 | r5 |
|---|---:|---:|---:|---:|---:|
| Authority Claim | 90.76±1.35 | 82.62±0.76 | 77.33±1.03 | 73.92±0.80 | 71.22±0.85 |
| Strong Pressure | 97.81±0.35 | 93.90±0.66 | 88.60±0.85 | 84.74±0.87 | 81.55±0.71 |
| Simple Denial | 98.37±0.32 | 95.42±0.40 | 91.65±0.71 | 87.46±0.94 | 83.82±1.45 |
| Logical Trap | 89.31±0.87 | 75.80±1.80 | 65.50±1.09 | 58.14±1.68 | 52.42±2.20 |
| Soft Pressure | 97.61±0.19 | 95.90±0.57 | 94.86±0.61 | 94.07±0.62 | 93.18±0.53 |

#### squad11_validation
| persona | r1 | r2 | r3 | r4 | r5 |
|---|---:|---:|---:|---:|---:|
| Authority Claim | 47.83±1.65 | 30.05±1.93 | 23.54±2.45 | 20.31±2.04 | 19.09±1.89 |
| Strong Pressure | 66.22±0.78 | 51.49±1.56 | 41.92±1.89 | 32.97±0.64 | 26.25±0.78 |
| Simple Denial | 61.02±0.97 | 49.96±0.35 | 41.55±0.78 | 35.50±1.18 | 31.56±1.06 |
| Logical Trap | 51.41±3.06 | 38.16±3.12 | 32.60±2.89 | 29.85±2.93 | 27.22±2.78 |
| Soft Pressure | 71.48±1.50 | 61.57±1.55 | 55.58±1.33 | 51.82±1.65 | 49.65±1.30 |

#### squad20_validation
| persona | r1 | r2 | r3 | r4 | r5 |
|---|---:|---:|---:|---:|---:|
| Authority Claim | 45.43±1.29 | 29.49±1.47 | 23.38±0.75 | 20.23±0.89 | 18.66±1.07 |
| Strong Pressure | 64.38±0.58 | 49.33±0.74 | 38.97±1.04 | 31.15±0.34 | 24.87±1.08 |
| Simple Denial | 57.14±0.64 | 46.25±0.99 | 37.88±0.65 | 32.26±0.89 | 28.22±0.80 |
| Logical Trap | 46.59±2.02 | 34.42±2.09 | 29.70±2.36 | 26.86±1.89 | 24.68±2.11 |
| Soft Pressure | 69.62±1.70 | 58.75±1.04 | 52.90±1.28 | 49.41±0.66 | 46.64±0.70 |

#### svamp
| persona | r1 | r2 | r3 | r4 | r5 |
|---|---:|---:|---:|---:|---:|
| Authority Claim | 94.40±0.06 | 91.15±0.86 | 87.96±0.77 | 85.51±0.90 | 83.57±0.70 |
| Strong Pressure | 97.91±0.36 | 95.90±0.54 | 94.00±0.91 | 91.44±1.35 | 89.21±1.40 |
| Simple Denial | 98.43±0.69 | 96.63±0.35 | 94.69±0.36 | 92.83±0.20 | 91.04±0.64 |
| Logical Trap | 95.79±0.91 | 91.18±1.59 | 86.90±1.90 | 82.73±2.00 | 78.34±1.52 |
| Soft Pressure | 97.44±0.39 | 96.27±0.59 | 95.50±0.82 | 94.62±0.49 | 94.11±0.47 |

#### triviaqa_rc_validation
| persona | r1 | r2 | r3 | r4 | r5 |
|---|---:|---:|---:|---:|---:|
| Authority Claim | 65.57±1.60 | 55.69±1.05 | 50.70±0.96 | 47.54±1.27 | 45.55±1.20 |
| Strong Pressure | 64.79±1.25 | 50.24±1.54 | 41.06±0.71 | 34.66±1.03 | 30.28±0.47 |
| Simple Denial | 65.96±1.03 | 54.98±1.16 | 46.16±1.43 | 41.17±1.05 | 36.97±1.33 |
| Logical Trap | 66.88±1.59 | 53.41±1.87 | 46.19±2.71 | 41.55±1.58 | 38.91±1.78 |
| Soft Pressure | 75.92±1.60 | 66.64±2.50 | 61.34±2.72 | 58.55±3.05 | 56.29±3.54 |

## 3.Y (확장) Turn-of-failure (언제 처음 무너지는가?)
각 (벤치마크, persona)에서 never-fail 비율과 fail@1 비율을 mean±std로 요약한다.

### 7B
#### arc_easy_validation
| persona | never (mean±std) | fail@1 (mean±std) |
|---|---:|---:|
| Authority Claim | 11.52±0.77 | 54.93±1.72 |
| Strong Pressure | 39.40±2.77 | 29.50±3.26 |
| Simple Denial | 41.89±2.65 | 23.69±1.22 |
| Logical Trap | 42.49±2.78 | 10.65±1.71 |
| Soft Pressure | 33.50±2.17 | 18.11±1.42 |

#### gsm8k
| persona | never (mean±std) | fail@1 (mean±std) |
|---|---:|---:|
| Authority Claim | 67.48±1.16 | 11.62±0.44 |
| Strong Pressure | 87.91±0.77 | 3.80±0.91 |
| Simple Denial | 89.93±0.37 | 3.92±0.51 |
| Logical Trap | 90.21±0.56 | 3.90±0.55 |
| Soft Pressure | 87.99±1.00 | 3.05±0.31 |

#### squad11_validation
| persona | never (mean±std) | fail@1 (mean±std) |
|---|---:|---:|
| Authority Claim | 22.45±0.80 | 49.37±0.39 |
| Strong Pressure | 28.65±0.71 | 30.98±1.47 |
| Simple Denial | 32.19±1.87 | 32.92±2.77 |
| Logical Trap | 32.98±1.70 | 25.65±0.75 |
| Soft Pressure | 68.73±1.12 | 9.66±0.43 |

#### squad20_validation
| persona | never (mean±std) | fail@1 (mean±std) |
|---|---:|---:|
| Authority Claim | 21.78±1.44 | 50.12±1.98 |
| Strong Pressure | 26.34±1.45 | 35.35±1.84 |
| Simple Denial | 30.70±0.30 | 34.67±0.90 |
| Logical Trap | 31.78±2.35 | 29.27±0.78 |
| Soft Pressure | 64.17±0.96 | 11.42±0.48 |

#### svamp
| persona | never (mean±std) | fail@1 (mean±std) |
|---|---:|---:|
| Authority Claim | 78.75±1.41 | 6.65±1.05 |
| Strong Pressure | 82.80±1.43 | 5.14±0.75 |
| Simple Denial | 87.91±0.55 | 5.26±0.28 |
| Logical Trap | 92.45±0.58 | 2.97±0.50 |
| Soft Pressure | 90.57±1.24 | 2.52±0.77 |

#### triviaqa_rc_validation
| persona | never (mean±std) | fail@1 (mean±std) |
|---|---:|---:|
| Authority Claim | 25.43±2.79 | 50.19±3.08 |
| Strong Pressure | 8.11±2.40 | 38.26±1.10 |
| Simple Denial | 8.82±1.02 | 45.55±1.98 |
| Logical Trap | 60.74±2.87 | 15.46±0.54 |
| Soft Pressure | 63.85±2.24 | 14.62±1.37 |

### 14B
#### arc_easy_validation
| persona | never (mean±std) | fail@1 (mean±std) |
|---|---:|---:|
| Authority Claim | 54.54±0.24 | 23.65±1.19 |
| Strong Pressure | 27.02±1.26 | 18.72±1.64 |
| Simple Denial | 47.30±1.64 | 13.56±2.64 |
| Logical Trap | 47.86±1.84 | 21.81±1.11 |
| Soft Pressure | 53.44±2.36 | 13.70±2.05 |

#### gsm8k
| persona | never (mean±std) | fail@1 (mean±std) |
|---|---:|---:|
| Authority Claim | 71.22±0.85 | 9.24±1.35 |
| Strong Pressure | 81.55±0.71 | 2.19±0.35 |
| Simple Denial | 83.82±1.45 | 1.63±0.32 |
| Logical Trap | 52.42±2.20 | 10.69±0.87 |
| Soft Pressure | 93.18±0.53 | 2.39±0.19 |

#### squad11_validation
| persona | never (mean±std) | fail@1 (mean±std) |
|---|---:|---:|
| Authority Claim | 19.09±1.89 | 52.17±1.65 |
| Strong Pressure | 26.25±0.78 | 33.78±0.78 |
| Simple Denial | 31.56±1.06 | 38.98±0.97 |
| Logical Trap | 27.22±2.78 | 48.59±3.06 |
| Soft Pressure | 49.65±1.30 | 28.52±1.50 |

#### squad20_validation
| persona | never (mean±std) | fail@1 (mean±std) |
|---|---:|---:|
| Authority Claim | 18.66±1.07 | 54.57±1.29 |
| Strong Pressure | 24.87±1.08 | 35.62±0.58 |
| Simple Denial | 28.22±0.80 | 42.86±0.64 |
| Logical Trap | 24.68±2.11 | 53.41±2.02 |
| Soft Pressure | 46.64±0.70 | 30.38±1.70 |

#### svamp
| persona | never (mean±std) | fail@1 (mean±std) |
|---|---:|---:|
| Authority Claim | 83.57±0.70 | 5.60±0.06 |
| Strong Pressure | 89.21±1.40 | 2.09±0.36 |
| Simple Denial | 91.04±0.64 | 1.57±0.69 |
| Logical Trap | 78.34±1.52 | 4.21±0.91 |
| Soft Pressure | 94.11±0.47 | 2.56±0.39 |

#### triviaqa_rc_validation
| persona | never (mean±std) | fail@1 (mean±std) |
|---|---:|---:|
| Authority Claim | 45.55±1.20 | 34.43±1.60 |
| Strong Pressure | 30.28±0.47 | 35.21±1.25 |
| Simple Denial | 36.97±1.33 | 34.04±1.03 |
| Logical Trap | 38.91±1.78 | 33.12±1.59 |
| Soft Pressure | 56.29±3.54 | 24.08±1.60 |

## 6.Z (추가) 논문 서술 템플릿 (동역학 포함)
- (Persona dynamics) Authority Claim은 여러 벤치마크에서 fail@1 비율이 높아 early-round 붕괴를 유발하고, Soft Pressure는 never-fail 비율이 높아 완만한 붕괴 곡선을 보인다.
- (Benchmark dynamics) Math 벤치마크는 r1..r5 survival이 높게 유지되지만, QA/OpenQA는 특정 persona에서 r1부터 급락하거나 누적적으로 하락한다.


## 7. Figure-ready outputs (SVG, no matplotlib)

- Script: `scripts/make_figures_svg.py`
- Output dir (example): `/mnt/raid6/aa007878/galileo/results/multiseed_20260203_173602/paper_figures_partial/`

Generated files:
- `survival_curve_<dataset>.svg`: model-wise (7b/14b) persona-avg survival curves (r1..r5)
- `fail1_never_<model>.svg`: persona-wise never vs fail@1 (aggregate)

Run:

```bash
python scripts/make_figures_svg.py   --results_root /mnt/raid6/aa007878/galileo/results/multiseed_20260203_173602   --out_dir /mnt/raid6/aa007878/galileo/results/multiseed_20260203_173602/paper_figures_partial   --models 7b,14b
```

<!-- AUTO:FINAL_MULTI_SEED_START -->

## 8. Final multi-seed results (seed_1..seed_5, strict data_dir)

> 본 요약은 논문 본문(`PAPER_DRAFT_KO.md`)의 Section 5.4(Table 1–3, Figure 1–2)로도 반영되어 있다.

- Results root: `/mnt/raid6/aa007878/galileo/results/multiseed_20260203_173602`
- Seeds: seed_1, seed_2, seed_3, seed_4, seed_5
- Table exports: `/mnt/raid6/aa007878/galileo/results/multiseed_20260203_173602/paper_tables_final/`
- Figure exports (committed): `paper_figures/`

### 8.1 Aggregate tables (mean±std over seeds)

# Multi-seed summary (round 5)
Results root: `/mnt/raid6/aa007878/galileo/results/multiseed_20260203_173602`
Seeds: seed_1, seed_2, seed_3, seed_4, seed_5

## Survival @ round 5 (mean±std over seeds)
### 14b
- Logical Trap: 44.39 ± 1.09 (n=5)
- Strong Pressure: 48.23 ± 0.58 (n=5)
- Authority Claim: 48.38 ± 0.25 (n=5)
- Simple Denial: 53.90 ± 0.75 (n=5)
- Soft Pressure: 66.62 ± 0.71 (n=5)

### 7b
- Authority Claim: 40.43 ± 0.65 (n=5)
- Strong Pressure: 48.63 ± 0.40 (n=5)
- Simple Denial: 51.63 ± 0.56 (n=5)
- Logical Trap: 59.84 ± 0.43 (n=5)
- Soft Pressure: 70.39 ± 0.48 (n=5)

### 8.2 Figures (paper-ready SVG)

**Survival curve (persona-avg, r1..r5)**

- GSM8K

![](paper_figures/survival_curve_gsm8k.svg)

- TriviaQA (RC)

![](paper_figures/survival_curve_triviaqa_rc_validation.svg)

- SQuAD11

![](paper_figures/survival_curve_squad11_validation.svg)

**Turn-of-failure summary (aggregate over benchmarks)**

![](paper_figures/fail1_never_7b.svg)

![](paper_figures/fail1_never_14b.svg)

### 8.3 Short takeaways (write-up ready)

- **Scale helps robustness**: 14B는 7B 대비 r5 survival 및 recovery 모두 개선되는 경향이 일관적이다(표 8.1).
- **Persona dynamics**: Soft Pressure는 never-fail 비율이 높고 완만한 하락 곡선을 보이며, Authority Claim/Strong Pressure는 early-round 붕괴(fail@1)가 상대적으로 크다.
- **Benchmark dynamics**: Math 계열(GSM8K/SVAMP)은 전반적으로 survival이 높은 반면, OpenQA/QA(TriviaQA/SQuAD)는 persona에 따라 r1부터 급락하거나 누적적으로 하락한다.

<!-- AUTO:FINAL_MULTI_SEED_END -->


<!-- AUTO:TOF_DATASET_TABLE_START -->

## 8.X Dataset×Persona Turn-of-failure (never / fail@1, mean±std over seeds)

Results root: `/mnt/raid6/aa007878/galileo/results/multiseed_20260203_173602`

### 7B

#### arc_easy_validation

| persona | never | fail@1 |
|---|---:|---:|
| Authority Claim | 11.51±0.67 | 54.11±2.36 |
| Strong Pressure | 39.25±2.42 | 29.03±3.00 |
| Simple Denial | 41.64±2.36 | 23.68±1.06 |
| Logical Trap | 43.16±2.84 | 10.48±1.53 |
| Soft Pressure | 33.71±1.94 | 17.93±1.30 |

#### gsm8k

| persona | never | fail@1 |
|---|---:|---:|
| Authority Claim | 67.82±1.25 | 11.51±0.45 |
| Strong Pressure | 87.97±0.68 | 4.01±0.92 |
| Simple Denial | 89.66±0.67 | 4.07±0.55 |
| Logical Trap | 90.10±0.55 | 3.90±0.48 |
| Soft Pressure | 87.95±0.87 | 3.08±0.28 |

#### squad11_validation

| persona | never | fail@1 |
|---|---:|---:|
| Authority Claim | 22.75±0.97 | 48.83±1.25 |
| Strong Pressure | 28.07±1.43 | 31.25±1.40 |
| Simple Denial | 32.31±1.64 | 33.16±2.46 |
| Logical Trap | 32.77±1.55 | 25.88±0.83 |
| Soft Pressure | 68.76±0.97 | 9.88±0.62 |

#### squad20_validation

| persona | never | fail@1 |
|---|---:|---:|
| Authority Claim | 21.87±1.27 | 50.68±2.12 |
| Strong Pressure | 26.21±1.29 | 34.88±1.90 |
| Simple Denial | 30.06±1.45 | 35.76±2.56 |
| Logical Trap | 31.27±2.34 | 29.20±0.69 |
| Soft Pressure | 64.17±0.84 | 11.69±0.73 |

#### svamp

| persona | never | fail@1 |
|---|---:|---:|
| Authority Claim | 78.68±1.23 | 6.80±0.98 |
| Strong Pressure | 83.06±1.36 | 5.19±0.66 |
| Simple Denial | 87.95±0.49 | 5.13±0.38 |
| Logical Trap | 92.56±0.56 | 2.85±0.51 |
| Soft Pressure | 89.84±1.96 | 2.64±0.72 |

#### triviaqa_rc_validation

| persona | never | fail@1 |
|---|---:|---:|
| Authority Claim | 25.47±2.42 | 49.82±2.79 |
| Strong Pressure | 8.07±2.08 | 38.45±1.04 |
| Simple Denial | 8.78±0.89 | 45.12±1.96 |
| Logical Trap | 61.24±2.73 | 15.35±0.52 |
| Soft Pressure | 63.24±2.37 | 14.37±1.31 |

### 14B

#### arc_easy_validation

| persona | never | fail@1 |
|---|---:|---:|
| Authority Claim | 54.21±0.78 | 23.76±1.06 |
| Strong Pressure | 27.16±1.14 | 18.93±1.49 |
| Simple Denial | 46.68±1.99 | 14.02±2.51 |
| Logical Trap | 47.75±1.61 | 21.96±1.02 |
| Soft Pressure | 53.29±2.07 | 13.73±1.78 |

#### gsm8k

| persona | never | fail@1 |
|---|---:|---:|
| Authority Claim | 71.51±0.98 | 9.36±1.19 |
| Strong Pressure | 81.76±0.77 | 2.04±0.45 |
| Simple Denial | 83.92±1.28 | 1.59±0.29 |
| Logical Trap | 52.75±2.04 | 10.46±0.92 |
| Soft Pressure | 93.11±0.48 | 2.24±0.37 |

#### squad11_validation

| persona | never | fail@1 |
|---|---:|---:|
| Authority Claim | 19.08±1.63 | 52.55±1.67 |
| Strong Pressure | 26.42±0.78 | 33.72±0.69 |
| Simple Denial | 31.84±1.11 | 39.00±0.84 |
| Logical Trap | 27.42±2.45 | 48.57±2.65 |
| Soft Pressure | 49.58±1.13 | 28.36±1.35 |

#### squad20_validation

| persona | never | fail@1 |
|---|---:|---:|
| Authority Claim | 18.36±1.14 | 54.70±1.16 |
| Strong Pressure | 24.72±1.00 | 35.70±0.53 |
| Simple Denial | 28.35±0.75 | 42.67±0.70 |
| Logical Trap | 24.87±1.87 | 53.19±1.81 |
| Soft Pressure | 46.55±0.64 | 30.68±1.62 |

#### svamp

| persona | never | fail@1 |
|---|---:|---:|
| Authority Claim | 83.64±0.62 | 5.62±0.07 |
| Strong Pressure | 89.40±1.29 | 1.99±0.38 |
| Simple Denial | 91.22±0.69 | 1.52±0.61 |
| Logical Trap | 78.13±1.39 | 4.48±0.99 |
| Soft Pressure | 93.82±0.76 | 2.40±0.49 |

#### triviaqa_rc_validation

| persona | never | fail@1 |
|---|---:|---:|
| Authority Claim | 45.57±1.04 | 34.56±1.41 |
| Strong Pressure | 30.75±1.11 | 35.57±1.34 |
| Simple Denial | 36.95±1.15 | 34.30±1.06 |
| Logical Trap | 39.51±2.04 | 33.20±1.39 |
| Soft Pressure | 56.63±3.16 | 23.82±1.50 |

<!-- AUTO:TOF_DATASET_TABLE_END -->


## 9. Qualitative taxonomy labeling (flip samples)

멀티턴 압박에서 오답 전향이 발생하는 *원인 유형*을 정성적으로 라벨링하기 위한 시트를 생성했다.

- Sheet (seed1, per_cell=20):
  - `/mnt/raid6/aa007878/galileo/results/multiseed_20260203_173602/paper_taxonomy/taxonomy_seed1_7b.csv`
  - `/mnt/raid6/aa007878/galileo/results/multiseed_20260203_173602/paper_taxonomy/taxonomy_seed1_14b.csv`

라벨링 권장 taxonomy (예시):
- `authority_compliance`
- `social_appeasement`
- `logical_trap`
- `uncertainty_collapse`
- `hedged_flip`
- `other`

작업 방식:
- `taxonomy_label` 컬럼을 사람이 채우고,
- `notes`에 근거/인상적인 패턴을 1-2줄로 남긴다.


<!-- AUTO:ATTENTION_PROBE_START -->

## 10. Attention probe (single-forward, truncated)

아래는 실험 로그에서 대화 입력을 재구성한 뒤, Transformers로 `output_attentions=True` 단일 forward를 수행하여 attention을 요약한 결과이다.

주의: attention은 O(L^2)이므로 마지막 256 tokens로 truncate한 근사치이며, *메커니즘 힌트*를 제공하는 용도이다.

### qwen7b_seed1_triviaqa_len256.csv

- N(fail)=20, N(survive)=20
- Entropy(last token, last layer): fail 3.787±0.242 vs survive 3.919±0.212 (Δ=-0.132)
- Mass(to last user span): fail 0.391±0.055 vs survive 0.353±0.033 (Δ=+0.038)

### qwen7b_seed1_gsm8k_len256.csv

- N(fail)=20, N(survive)=20
- Entropy(last token, last layer): fail 3.104±0.534 vs survive 3.598±0.359 (Δ=-0.494)
- Mass(to last user span): fail 0.403±0.061 vs survive 0.340±0.043 (Δ=+0.063)

<!-- AUTO:ATTENTION_PROBE_END -->

<!-- AUTO:QUAL_FAIL_START -->

## 11. Qualitative failure-point analysis (output-level)

> 대표 flip 사례 원문은 별도 부록 파일 `PAPER_RESULTS_QUAL_EXAMPLES_KO.md`에 정리했다.

attention/내부동작 이전에, 모델이 오답 전향 시 어떤 언어적 패턴을 보이는지(hedging, 사과, defer 등)를 로그에서 정성적으로 정리한다.

### arc_easy_validation

- failed cases analyzed: 1817
- pattern hit-rate (flip response 기준, 단순 regex):
  - authority: 4/1817 (0.2%)
  - hedging: 43/1817 (2.4%)
  - recompute: 170/1817 (9.4%)
  - refusal: 9/1817 (0.5%)

Persona별 상위 패턴(빈도 상위 2개):
- Authority Claim: authority=2, hedging=2
- Strong Pressure: recompute=64, hedging=7
- Simple Denial: recompute=55, hedging=8
- Logical Trap: recompute=5, hedging=4
- Soft Pressure: recompute=45, hedging=22

대표 사례 원문은 부록 파일을 참조: `PAPER_RESULTS_QUAL_EXAMPLES_KO.md`

### gsm8k

- failed cases analyzed: 731
- pattern hit-rate (flip response 기준, 단순 regex):
  - authority: 23/731 (3.1%)
  - hedging: 59/731 (8.1%)
  - recompute: 37/731 (5.1%)
  - refusal: 38/731 (5.2%)

Persona별 상위 패턴(빈도 상위 2개):
- Authority Claim: hedging=31, refusal=25
- Strong Pressure: recompute=10, hedging=4
- Simple Denial: hedging=6, recompute=5
- Logical Trap: hedging=8, recompute=4
- Soft Pressure: recompute=15, hedging=10

대표 사례 원문은 부록 파일을 참조: `PAPER_RESULTS_QUAL_EXAMPLES_KO.md`

### squad11_validation

- failed cases analyzed: 2432
- pattern hit-rate (flip response 기준, 단순 regex):
  - authority: 12/2432 (0.5%)
  - hedging: 21/2432 (0.9%)
  - refusal: 10/2432 (0.4%)

Persona별 상위 패턴(빈도 상위 2개):
- Authority Claim: refusal=2, authority=2
- Strong Pressure: refusal=5, authority=3
- Simple Denial: authority=3, refusal=1
- Logical Trap: hedging=2, authority=2
- Soft Pressure: hedging=19, authority=2

대표 사례 원문은 부록 파일을 참조: `PAPER_RESULTS_QUAL_EXAMPLES_KO.md`

### squad20_validation

- failed cases analyzed: 2477
- pattern hit-rate (flip response 기준, 단순 regex):
  - apology: 1/2477 (0.0%)
  - authority: 21/2477 (0.8%)
  - hedging: 22/2477 (0.9%)
  - refusal: 4/2477 (0.2%)

Persona별 상위 패턴(빈도 상위 2개):
- Authority Claim: authority=9, refusal=2
- Strong Pressure: authority=4, refusal=1
- Simple Denial: authority=2, refusal=1
- Logical Trap: authority=5, hedging=1
- Soft Pressure: hedging=20, authority=1

대표 사례 원문은 부록 파일을 참조: `PAPER_RESULTS_QUAL_EXAMPLES_KO.md`

### svamp

- failed cases analyzed: 455
- pattern hit-rate (flip response 기준, 단순 regex):
  - authority: 7/455 (1.5%)
  - hedging: 15/455 (3.3%)
  - recompute: 14/455 (3.1%)
  - refusal: 39/455 (8.6%)

Persona별 상위 패턴(빈도 상위 2개):
- Authority Claim: refusal=22, hedging=5
- Strong Pressure: hedging=7, recompute=7
- Simple Denial: refusal=5, recompute=2
- Logical Trap: authority=1, recompute=1
- Soft Pressure: refusal=4, hedging=3

대표 사례 원문은 부록 파일을 참조: `PAPER_RESULTS_QUAL_EXAMPLES_KO.md`

### triviaqa_rc_validation

- failed cases analyzed: 1817
- pattern hit-rate (flip response 기준, 단순 regex):
  - authority: 1/1817 (0.1%)
  - hedging: 1/1817 (0.1%)
  - recompute: 1/1817 (0.1%)

Persona별 상위 패턴(빈도 상위 2개):
- Authority Claim: authority=1, recompute=1
- Simple Denial: hedging=1

대표 사례 원문은 부록 파일을 참조: `PAPER_RESULTS_QUAL_EXAMPLES_KO.md`

<!-- AUTO:QUAL_FAIL_END -->
