# 실험 결과 분석 (논문용)

> 아래 내용은 strict data_dir + multi-seed 결과의 **현재까지 누적분**으로 작성되었다.
> Results root: `/mnt/raid6/aa007878/galileo/results/multiseed_20260203_173602`
> 사용 가능한 seeds: 7B=['seed_1', 'seed_2', 'seed_3', 'seed_4'] / 14B=['seed_1', 'seed_2', 'seed_3']

## 1. 실험 설정
- 모델: Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct
- 벤치마크: gsm8k, svamp, arc_easy_validation, squad11_validation, squad20_validation, triviaqa_rc_validation
- persona: 5종(Soft/Denial/Strong/Authority/Trap), 최대 5라운드
- 보고 지표: initial accuracy / round별 survival / flip 이후 recovery

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
