# Qwen7B GSM8K comparable mains — 2026-03-11

Purpose: record the **single-dataset GSM8K** comparable main runs that are now available for:

- evidence-bearing recovery
- grounded correction
- evidence-gate retry + evidence-bearing recovery

All rows below are backed by validated `paper_exports/`.

---

## Result roots

- evidence-bearing:
  - `tmp/results/qwen7b_gsm8k_control_authority_evidence_gpu5_20260311_010641/`
- grounded correction:
  - `tmp/results/qwen7b_gsm8k_control_authority_grounded_gpu5_20260310_221155/`
- evidence-gate:
  - `tmp/results/qwen7b_gsm8k_control_authority_evidencegate_gpu5_20260310_230240/`

Validator status:

- all three pass `python3 scripts/validate_paper_exports.py --results_root <ROOT> --check_runner_parity`

---

## Comparable metrics

| variant | control Survival@5 | authority Survival@5 | authority Recovery@flip | notes |
|---|---:|---:|---:|---|
| evidence-bearing | 87.80% | 63.41% | 100.00% | newly completed on 2026-03-11 |
| grounded correction | 92.68% | 63.41% | 93.33% | validated 2026-03-10 |
| evidence-gate | 90.24% | 70.73% | 91.67% | validated 2026-03-10 |

Source files:

- evidence-bearing:
  - `tmp/results/qwen7b_gsm8k_control_authority_evidence_gpu5_20260311_010641/paper_exports/survival_curve.csv`
  - `tmp/results/qwen7b_gsm8k_control_authority_evidence_gpu5_20260311_010641/paper_exports/recovery_accuracy.csv`
- grounded correction:
  - `tmp/results/qwen7b_gsm8k_control_authority_grounded_gpu5_20260310_221155/paper_exports/survival_curve.csv`
  - `tmp/results/qwen7b_gsm8k_control_authority_grounded_gpu5_20260310_221155/paper_exports/recovery_accuracy.csv`
- evidence-gate:
  - `tmp/results/qwen7b_gsm8k_control_authority_evidencegate_gpu5_20260310_230240/paper_exports/survival_curve.csv`
  - `tmp/results/qwen7b_gsm8k_control_authority_evidencegate_gpu5_20260310_230240/paper_exports/recovery_accuracy.csv`

---

## Immediate interpretation guardrail

- This summary improves **single-dataset GSM8K comparability** across the three correction/retry variants.
- It does **not** by itself justify a broader mitigation claim beyond the validated Qwen7B package.
- ARC evidence-bearing already exists as a validated non-math direct-comparison root under:
  - `tmp/results/main_arc_gpu6_20260310_191906/`

## Execution note

- The first launch attempt from the sandbox failed before inference with a CUDA/vLLM device-detection error.
- A single escalated retry using the exact same experiment command succeeded and produced the validated result root above.
