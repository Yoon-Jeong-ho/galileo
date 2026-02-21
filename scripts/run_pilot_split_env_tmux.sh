#!/usr/bin/env bash
set -euo pipefail

# Split-env pilot runner for cross-family feasibility checks.
#
# Motivation: some nlp8 conda envs have vLLM (needed for run_experiment.py) but
# miss pandas deps (needed for paper_export.py / validators). This script runs:
#  (1) run_experiment.py in RUN_ENV (vLLM-capable)
#  (2) paper_export.py + write_runner_metadata.py + validate_paper_exports.py in EXPORT_ENV
#
# SSOT remote: nlp8, repo: /data_x/aa007878/galileo
#
# Usage (on nlp8):
#   GPU_LIST=1 RUN_ENV=UltraToolOpen EXPORT_ENV=emp \
#   MODEL=google/gemma-1.1-7b-it SEED=1 NUM_SAMPLES=50 MAX_MODEL_LEN=8192 MAX_TOKENS=2048 \
#   OUT=results/tier1_gemma1p1_7b_it_pilot_seed1_$(date +%Y%m%d_%H%M%S) \
#   bash scripts/run_pilot_split_env_tmux.sh gemma_pilot_s1

SESSION=${1:-galileo-pilot}

GPU_LIST=${GPU_LIST:-4}
TP_SIZE=${TP_SIZE:-1}

MODEL=${MODEL:-google/gemma-1.1-7b-it}
SEED=${SEED:-1}
NUM_SAMPLES=${NUM_SAMPLES:-50}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_TOKENS=${MAX_TOKENS:-2048}

# Envs
CONDA_BIN=${CONDA_BIN:-/mnt/raid6/aa007878/miniconda3/bin/conda}
RUN_ENV=${RUN_ENV:-UltraToolOpen}
EXPORT_ENV=${EXPORT_ENV:-emp}

DATA_ALL_DIR=${DATA_ALL_DIR:-/data_x/aa007878/galileo/data_all_strict}
MATH_DIR=${MATH_DIR:-/data_x/aa007878/galileo/data}
QA_DIR=${QA_DIR:-/data_x/aa007878/galileo/data_qa_full}

OUT=${OUT:-/data_x/aa007878/galileo/results/pilot_$(date +%Y%m%d_%H%M%S)}

mkdir -p "$OUT"

# Build strict unified data dir
MATH_DIR="$MATH_DIR" QA_DIR="$QA_DIR" bash scripts/make_all_data_dir_strict.sh "$DATA_ALL_DIR" >/dev/null

RUNNER="$OUT/run_pilot_split_env.sh"

cat > "$RUNNER" <<RUN1
#!/usr/bin/env bash
set -euo pipefail
cd /data_x/aa007878/galileo

GPU_LIST="$GPU_LIST"
TP_SIZE="$TP_SIZE"
MODEL="$MODEL"
SEED="$SEED"
NUM_SAMPLES="$NUM_SAMPLES"
MAX_MODEL_LEN="$MAX_MODEL_LEN"
MAX_TOKENS="$MAX_TOKENS"
RUN_ENV="$RUN_ENV"
EXPORT_ENV="$EXPORT_ENV"
CONDA_BIN="$CONDA_BIN"
DATA_ALL_DIR="$DATA_ALL_DIR"
OUT="$OUT"
RUN1

cat >> "$RUNNER" <<'RUN2'

echo "=== Pilot start: $(date) ===" | tee -a "$OUT/run.log"
echo "MODEL=$MODEL" | tee -a "$OUT/run.log"
echo "GPU_LIST=$GPU_LIST TP=$TP_SIZE" | tee -a "$OUT/run.log"
echo "NUM_SAMPLES=$NUM_SAMPLES MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_TOKENS=$MAX_TOKENS" | tee -a "$OUT/run.log"
echo "RUN_ENV=$RUN_ENV EXPORT_ENV=$EXPORT_ENV" | tee -a "$OUT/run.log"

export PYTHONUNBUFFERED=1

# Phase 1: run
CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  stdbuf -oL -eL "$CONDA_BIN" run -n "$RUN_ENV" python run_experiment.py \
    --model "$MODEL" \
    --data_dir "$DATA_ALL_DIR" \
    --results_dir "$OUT" \
    --tensor_parallel_size "$TP_SIZE" \
    --num_samples "$NUM_SAMPLES" \
    --seed "$SEED" \
    --max_model_len "$MAX_MODEL_LEN" \
    --max_tokens "$MAX_TOKENS" \
    2>&1 | tee -a "$OUT/run.log"

# Fail-fast: detect pathological caps
python3 scripts/check_runlog_for_token_caps.py "$OUT/run.log" 2>&1 | tee -a "$OUT/run.log"

# Phase 2: export + validate (env with pandas)
"$CONDA_BIN" run -n "$EXPORT_ENV" python scripts/paper_export.py \
  --results_root "$OUT" \
  --model_dir "$OUT/${MODEL##*/}" \
  --out_dir "$OUT/paper_exports" \
  --num_flip_samples 200 \
  --seed "$SEED" \
  2>&1 | tee -a "$OUT/run.log"

python3 scripts/write_runner_metadata.py \
  --paper_exports "$OUT/paper_exports" \
  --model "$MODEL" \
  --seed "$SEED" \
  --gpu_list "$GPU_LIST" \
  --tp "$TP_SIZE" \
  --num_samples "$NUM_SAMPLES" \
  --max_model_len "$MAX_MODEL_LEN" \
  --max_tokens "$MAX_TOKENS" \
  --conda_env "$RUN_ENV" \
  --extra_json "{\"export_env\": \"$EXPORT_ENV\", \"run_env\": \"$RUN_ENV\"}" \
  2>&1 | tee -a "$OUT/run.log"

"$CONDA_BIN" run -n "$EXPORT_ENV" python scripts/validate_paper_exports.py --results_root "$OUT" \
  2>&1 | tee -a "$OUT/run.log"

echo "=== Pilot done: $(date) ===" | tee -a "$OUT/run.log"
RUN2

chmod +x "$RUNNER"

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION" -n main
fi

tmux send-keys -t "${SESSION}:main" "$RUNNER" C-m

echo "Started tmux session: $SESSION"
echo "OUT: $OUT"
echo "Attach: tmux attach -t $SESSION"
