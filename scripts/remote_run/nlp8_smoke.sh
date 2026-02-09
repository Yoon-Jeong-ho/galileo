#!/usr/bin/env bash
set -euo pipefail

# Nlp8 smoke run: minimal end-to-end check
# - uses env python directly (avoid conda run/activate hooks)
# - writes paper_exports + metadata + runner_metadata
# - runs validator at the end

GPU=${GPU:-0}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
DATA_DIR=${DATA_DIR:-/data_x/aa007878/galileo/data_all_strict}
NUM_SAMPLES=${NUM_SAMPLES:-40}
SEED=${SEED:-1}
TP=${TP:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_TOKENS=${MAX_TOKENS:-1024}

REPO_DIR=${REPO_DIR:-/data_x/aa007878/galileo}
PY=${PY:-/data_x/aa007878/miniconda3/envs/galileo/bin/python}

cd "$REPO_DIR"
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-results/smoke_${TS}}

mkdir -p "$OUT"

echo "=== smoke start $(date) ===" | tee -a "$OUT/run.log"
echo "GPU=$GPU MODEL=$MODEL NUM_SAMPLES=$NUM_SAMPLES SEED=$SEED TP=$TP" | tee -a "$OUT/run.log"

# Ensure env binaries (e.g., ninja) are visible. This matters because we don't "conda activate".
ENV_BIN_DIR="$(dirname "$PY")"
export PATH="$ENV_BIN_DIR:$PATH"

env CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  "$PY" run_experiment.py \
    --model "$MODEL" \
    --data_dir "$DATA_DIR" \
    --results_dir "$OUT" \
    --tensor_parallel_size "$TP" \
    --num_samples "$NUM_SAMPLES" \
    --seed "$SEED" \
    --max_model_len "$MAX_MODEL_LEN" \
    --max_tokens "$MAX_TOKENS" \
  2>&1 | tee -a "$OUT/run.log"

# paper exports
PYTHONUNBUFFERED=1 \
  "$PY" scripts/paper_export.py \
    --results_root "$OUT" \
    --model_dir "$OUT/${MODEL##*/}" \
    --out_dir "$OUT/paper_exports" \
    --num_flip_samples 50 \
    --seed "$SEED" \
  2>&1 | tee -a "$OUT/run.log"

# runner metadata
cat > "$OUT/paper_exports/runner_metadata.json" <<JSON
{
  "generated_at": "$(date -Iseconds)",
  "gpu_list": "${GPU}",
  "tensor_parallel_size": ${TP},
  "num_samples": ${NUM_SAMPLES},
  "max_model_len": ${MAX_MODEL_LEN},
  "max_tokens": ${MAX_TOKENS},
  "conda_env": "galileo",
  "model": "${MODEL}",
  "seed": ${SEED},
  "tag": "smoke"
}
JSON

# validate
PYTHONUNBUFFERED=1 \
  "$PY" scripts/validate_paper_exports.py \
    --results_root "$OUT" \
    --check_runner_parity \
  2>&1 | tee -a "$OUT/run.log"

echo "=== smoke done $(date) ===" | tee -a "$OUT/run.log"
echo "OUT=$OUT"
