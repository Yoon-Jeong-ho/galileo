#!/usr/bin/env bash
set -euo pipefail

# QA pilot runner (ARC MCQA + SQuAD QA) in tmux.
# Uses galileo conda env.
# Default GPUs: 4,5,6,7 (to avoid clashing with math run on 0,1,2,3).

SESSION=${1:-galileo-qa-pilot}

GPU_LIST=${GPU_LIST:-4,5,6,7}
TP_SIZE=${TP_SIZE:-4}
DATA_DIR=${DATA_DIR:-/data_x/aa007878/galileo/data_qa_pilot}
RESULTS_ROOT=${RESULTS_ROOT:-/mnt/raid6/aa007878/galileo/results/qa_pilot_$(date +%Y%m%d_%H%M%S)}
NUM_SAMPLES=${NUM_SAMPLES:-100}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_TOKENS=${MAX_TOKENS:-2048}

CONDA_ENV=${CONDA_ENV:-galileo}
CONDA_BIN=${CONDA_BIN:-/data_x/aa007878/miniconda3/bin/conda}

MODEL_7B=${MODEL_7B:-Qwen/Qwen2.5-7B-Instruct}
MODEL_14B=${MODEL_14B:-Qwen/Qwen2.5-14B-Instruct}
MODEL_32B=${MODEL_32B:-deepseek-ai/DeepSeek-R1-Distill-Qwen-32B}

mkdir -p "$RESULTS_ROOT"

RUNNER="$RESULTS_ROOT/run_all.sh"
cat > "$RUNNER" <<RUN
#!/usr/bin/env bash
set -euo pipefail
cd /mnt/raid6/aa007878/galileo-dev

echo "=== Galileo QA pilot start: \$(date) ==="
echo "GPUs: $GPU_LIST / TP: $TP_SIZE"
echo "DATA_DIR: $DATA_DIR"
echo "RESULTS_ROOT: $RESULTS_ROOT"
echo "NUM_SAMPLES: $NUM_SAMPLES, MAX_MODEL_LEN: $MAX_MODEL_LEN, MAX_TOKENS: $MAX_TOKENS"

run_one() {
  local model="\$1"
  local tag="\$2"

  mkdir -p "$RESULTS_ROOT/\$tag"
  echo "[\$(date)] Starting: \$model (\$tag)" | tee -a "$RESULTS_ROOT/\$tag/run.log"

  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  "$CONDA_BIN" run -n "$CONDA_ENV" python run_experiment.py \
    --model "\$model" \
    --data_dir "$DATA_DIR" \
    --results_dir "$RESULTS_ROOT/\$tag" \
    --tensor_parallel_size "$TP_SIZE" \
    --num_samples "$NUM_SAMPLES" \
    --max_model_len "$MAX_MODEL_LEN" \
    --max_tokens "$MAX_TOKENS" \
    2>&1 | tee -a "$RESULTS_ROOT/\$tag/run.log"
}

run_one $MODEL_7B  7b

if [ -n ${MODEL_14B:-} ]; then
  run_one $MODEL_14B 14b
else
  echo [skip] MODEL_14B is empty
fi

if [ -n ${MODEL_32B:-} ]; then
  run_one $MODEL_32B 32b
else
  echo [skip] MODEL_32B is empty
fi

echo "=== Galileo QA pilot done: \$(date) ==="
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
