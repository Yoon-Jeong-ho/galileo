#!/usr/bin/env bash
set -euo pipefail

SESSION=${1:-galileo-multiseed}

GPU_LIST=${GPU_LIST:-4,5,6,7}
TP_SIZE=${TP_SIZE:-4}

DATA_ALL_DIR=${DATA_ALL_DIR:-/data_x/aa007878/galileo/data_all_strict}
MATH_DIR=${MATH_DIR:-/data_x/aa007878/galileo/data}
QA_DIR=${QA_DIR:-/data_x/aa007878/galileo/data_qa_full}

RESULTS_ROOT=${RESULTS_ROOT:-/mnt/raid6/aa007878/galileo/results/multiseed_$(date +%Y%m%d_%H%M%S)}
NUM_SAMPLES=${NUM_SAMPLES:-1000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_TOKENS=${MAX_TOKENS:-2048}
SEEDS=${SEEDS:-1,2,3,4,5}

CONDA_BIN=${CONDA_BIN:-/data_x/aa007878/miniconda3/bin/conda}
CONDA_ENV=${CONDA_ENV:-galileo}

MODEL_7B=${MODEL_7B:-Qwen/Qwen2.5-7B-Instruct}
MODEL_14B=${MODEL_14B:-Qwen/Qwen2.5-14B-Instruct}

mkdir -p "$RESULTS_ROOT"

# Build strict data dir (no legacy/pilot *_val_50 files)
MATH_DIR="$MATH_DIR" QA_DIR="$QA_DIR" bash scripts/make_all_data_dir_strict.sh "$DATA_ALL_DIR" >/dev/null

RUNNER="$RESULTS_ROOT/run_multiseed.sh"

# Write header with concrete config values
cat > "$RUNNER" <<RUN1
#!/usr/bin/env bash
set -euo pipefail
cd /mnt/raid6/aa007878/galileo-dev

GPU_LIST="$GPU_LIST"
TP_SIZE="$TP_SIZE"
DATA_ALL_DIR="$DATA_ALL_DIR"
RESULTS_ROOT="$RESULTS_ROOT"
NUM_SAMPLES="$NUM_SAMPLES"
MAX_MODEL_LEN="$MAX_MODEL_LEN"
MAX_TOKENS="$MAX_TOKENS"
SEEDS="$SEEDS"
CONDA_BIN="$CONDA_BIN"
CONDA_ENV="$CONDA_ENV"
MODEL_7B="$MODEL_7B"
MODEL_14B="$MODEL_14B"

RUN1

# Write the actual runner logic without expansion
cat >> "$RUNNER" <<"RUN2"

echo "=== Galileo multi-seed start: $(date) ==="
echo "GPUs: ${GPU_LIST} / TP: ${TP_SIZE}"
echo "DATA_ALL_DIR: ${DATA_ALL_DIR}"
echo "RESULTS_ROOT: ${RESULTS_ROOT}"
echo "NUM_SAMPLES: ${NUM_SAMPLES}, MAX_MODEL_LEN: ${MAX_MODEL_LEN}, MAX_TOKENS: ${MAX_TOKENS}"
echo "SEEDS: ${SEEDS}"

to_array() {
  local s="$1"
  IFS=, read -r -a arr <<< "$s"
  echo "${arr[@]}"
}

run_one() {
  local seed="$1"
  local model="$2"
  local tag="$3"
  local out_dir="${RESULTS_ROOT}/seed_${seed}/${tag}"

  mkdir -p "$out_dir"
  echo "[$(date)] seed=${seed} model=${model} tag=${tag}" | tee -a "$out_dir/run.log"

  CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python run_experiment.py \
    --model "${model}" \
    --data_dir "${DATA_ALL_DIR}" \
    --results_dir "${out_dir}" \
    --tensor_parallel_size "${TP_SIZE}" \
    --num_samples "${NUM_SAMPLES}" \
    --seed "${seed}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --max_tokens "${MAX_TOKENS}" \
    2>&1 | tee -a "$out_dir/run.log"

  python scripts/paper_export.py \
    --results_root "${out_dir}" \
    --model_dir "${out_dir}/${model##*/}" \
    --out_dir "${out_dir}/paper_exports" \
    --num_flip_samples 200 \
    --seed "${seed}" \
    2>&1 | tee -a "$out_dir/run.log"

  # Runner-side metadata (auditable run settings). Keep separate from paper_export.py metadata.
  cat > "${out_dir}/paper_exports/runner_metadata.json" <<JSON
{
  "generated_at": "$(date -Iseconds)",
  "gpu_list": "${GPU_LIST}",
  "tensor_parallel_size": ${TP_SIZE},
  "num_samples": ${NUM_SAMPLES},
  "max_model_len": ${MAX_MODEL_LEN},
  "max_tokens": ${MAX_TOKENS},
  "conda_env": "${CONDA_ENV}",
  "model": "${model}",
  "seed": ${seed}
}
JSON

  # Fail fast if exports are incomplete.
  python scripts/validate_paper_exports.py --results_root "${out_dir}" \
    2>&1 | tee -a "$out_dir/run.log"
}

seeds=( $(to_array "${SEEDS}") )

for seed in "${seeds[@]}"; do
  run_one "${seed}" "${MODEL_7B}"  "7b"
  run_one "${seed}" "${MODEL_14B}" "14b"
done

echo "=== Galileo multi-seed done: $(date) ==="
RUN2

chmod +x "$RUNNER"

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION" -n main
fi

tmux send-keys -t "${SESSION}:main" "$RUNNER" C-m

echo "Started tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Results: $RESULTS_ROOT"
echo "Runner: $RUNNER"
