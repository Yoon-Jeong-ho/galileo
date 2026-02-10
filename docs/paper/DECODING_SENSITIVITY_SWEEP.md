# Decoding sensitivity sweep (Tier‑1 plan)

Goal: demonstrate that the main GALILEO findings (persona pressure vs Neutral Re‑asking Control) are not an artifact of one decoding configuration.

Scope (minimal, reviewer-risk focused):
- Model: **Qwen/Qwen2.5-7B-Instruct**
- Seeds: **1–2** (Tier‑1)
- Samples: **80** per seed (match main baseline)
- Results root: **`results_paper/`** (paper SSOT)
- Sweep variable: **`--greedy_temperature`** for adversarial+recovery turns.
  - Note: initial evaluation uses beam search; this sweep targets the multi-turn phase where pressure accumulates.

## Proposed settings

| setting | greedy_temperature | rationale |
|---|---:|---|
| T0 | 0.0 | deterministic multi-turn behavior (reduces randomness) |
| T1 | 0.7 | moderate sampling (stress-test robustness of qualitative pattern) |

## Expected outputs (paper-ready)
Each run must produce:
- `paper_exports/` with `survival_curve.csv`, `turn_of_failure.csv`, `flip_samples.csv`, `metadata.json`, `runner_metadata.json`
- validator prints `[OK] .../paper_exports` and `[OK] runner_metadata parity`

## Remote run commands (nlp8)

Hard SSOT: use **nlp8** repo `/data_x/aa007878/galileo` and GPUs **4/5/6** only.

### 0) Pre-check
```bash
ssh nlp8
cd /data_x/aa007878/galileo
nvidia-smi -i 4,5,6
python3 scripts/validate_paper_exports.py --results_root results_paper --check_runner_parity
```

### 1) Launch runs (tmux)
We recommend **one run per GPU**.

Template (fill GPU/SEED/TAG/TEMP):
```bash
GPU=4 SEED=1 TEMP=0.0 TAG=qwen_temp0_seed1 \
  OUT=results_paper/${TAG} \
  bash -lc 'cd /data_x/aa007878/galileo && \
  PY=/data_x/aa007878/miniconda3/envs/galileo/bin/python \
  CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY run_experiment.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data_dir /data_x/aa007878/galileo/data_all_strict \
    --results_dir '$OUT' \
    --num_samples 80 \
    --seed '$SEED' \
    --greedy_temperature '$TEMP' \
  2>&1 | tee -a '$OUT'/run.log'
```

Then generate paper exports + runner metadata + validate (same as existing paper-ready runs):
```bash
PY=/data_x/aa007878/miniconda3/envs/galileo/bin/python
OUT=results_paper/qwen_temp0_seed1
MODEL=Qwen/Qwen2.5-7B-Instruct
SEED=1
GPU=4

PYTHONUNBUFFERED=1 $PY scripts/paper_export.py \
  --results_root $OUT \
  --model_dir $OUT/${MODEL##*/} \
  --out_dir $OUT/paper_exports \
  --num_flip_samples 50 \
  --seed $SEED 2>&1 | tee -a $OUT/run.log

cat > $OUT/paper_exports/runner_metadata.json <<JSON
{
  "generated_at": "$(date -Iseconds)",
  "gpu_list": "${GPU}",
  "tensor_parallel_size": 1,
  "num_samples": 80,
  "greedy_temperature": ${TEMP},
  "model": "${MODEL}",
  "seed": ${SEED},
  "tag": "decoding_sweep"
}
JSON

PYTHONUNBUFFERED=1 $PY scripts/validate_paper_exports.py \
  --results_root $OUT \
  --check_runner_parity 2>&1 | tee -a $OUT/run.log
```

### 2) Paper SSOT global validation
After all sweep runs:
```bash
python3 scripts/validate_paper_exports.py --results_root results_paper --check_runner_parity \
  2>&1 | tee results_paper/GLOBAL_VALIDATE.log
```

## Notes / pitfalls
- Keep `results_paper/` clean: only put **citeable** runs here.
- Ensure runner metadata includes `greedy_temperature` so parity checks don’t incorrectly group mismatched settings.
- If the run is too heavy, reduce to **1 seed** first (smoke), then fill seed2.
