#!/usr/bin/env bash
set -euo pipefail

# Run math + QA + MCQA datasets together in ONE run_experiment.py invocation.
# Plan A: create a unified data_dir and pass it to --data_dir.

SESSION=${1:-galileo-all}

GPU_LIST=${GPU_LIST:-4,5,6,7}
TP_SIZE=${TP_SIZE:-4}

DATA_ALL_DIR=${DATA_ALL_DIR:-/data_x/aa007878/galileo/data_all_strict}
MATH_DIR=${MATH_DIR:-/data_x/aa007878/galileo/data}
QA_DIR=${QA_DIR:-/data_x/aa007878/galileo/data_qa_full}

RESULTS_ROOT=${RESULTS_ROOT:-/mnt/raid6/aa007878/galileo/results/all_pilot_$(date +%Y%m%d_%H%M%S)}
NUM_SAMPLES=${NUM_SAMPLES:-100}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_TOKENS=${MAX_TOKENS:-2048}
SEED=${SEED:-42}

CONDA_BIN=${CONDA_BIN:-/data_x/aa007878/miniconda3/bin/conda}
CONDA_ENV=${CONDA_ENV:-galileo}

MODEL_7B=${MODEL_7B:-Qwen/Qwen2.5-7B-Instruct}
MODEL_14B=${MODEL_14B:-Qwen/Qwen2.5-14B-Instruct}
MODEL_32B=${MODEL_32B:-deepseek-ai/DeepSeek-R1-Distill-Qwen-32B}

mkdir -p "$RESULTS_ROOT"

MATH_DIR="$MATH_DIR" QA_DIR="$QA_DIR" bash scripts/make_all_data_dir.sh "$DATA_ALL_DIR"

RUNNER="$RESULTS_ROOT/run_all.sh"
cat > "$RUNNER" <<RUN
#!/usr/bin/env bash
set -euo pipefail
cd /mnt/raid6/aa007878/galileo-dev

echo "=== Galileo ALL pilot start: \$(date) ==="
echo "GPUs: $GPU_LIST / TP: $TP_SIZE"
echo "DATA_ALL_DIR: $DATA_ALL_DIR"
echo "RESULTS_ROOT: $RESULTS_ROOT"
echo "NUM_SAMPLES: $NUM_SAMPLES, MAX_MODEL_LEN: $MAX_MODEL_LEN, MAX_TOKENS: $MAX_TOKENS"
echo "SEED: $SEED"

run_one() {
  local model="\$1"
  local tag="\$2"

  mkdir -p "$RESULTS_ROOT/\$tag"
  echo "[\$(date)] Starting: \$model (\$tag)" | tee -a "$RESULTS_ROOT/\$tag/run.log"

  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  "$CONDA_BIN" run -n "$CONDA_ENV" python run_experiment.py \
    --model "\$model" \
    --data_dir "$DATA_ALL_DIR" \
    --results_dir "$RESULTS_ROOT/\$tag" \
    --tensor_parallel_size "$TP_SIZE" \
    --num_samples "$NUM_SAMPLES" \
    --seed "$SEED" \
    --max_model_len "$MAX_MODEL_LEN" \
    --max_tokens "$MAX_TOKENS" \
    2>&1 | tee -a "$RESULTS_ROOT/\$tag/run.log"
}

run_one "$MODEL_7B"  "7b"

if [ -n "${MODEL_14B:-}" ]; then
  run_one "$MODEL_14B" "14b"
else
  echo "[skip] MODEL_14B is empty"
fi

if [ -n "${MODEL_32B:-}" ]; then
  run_one "$MODEL_32B" "32b"
else
  echo "[skip] MODEL_32B is empty"
fi

echo "=== Galileo ALL pilot done: \$(date) ==="
RUN
chmod +x "$RUNNER"

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION" -n main
fi

tmux send-keys -t "${SESSION}:main" "$RUNNER" C-m

echo "Started tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Results: $RESULTS_ROOT"
echo "Runner: $RUNNER"
