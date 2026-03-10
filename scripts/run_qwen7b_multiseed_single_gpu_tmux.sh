#!/usr/bin/env bash
set -euo pipefail

SESSION=${1:-qwen7b-multiseed}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=${REPO_DIR:-"$(cd "${SCRIPT_DIR}/.." && pwd)"}

GPU=${GPU:-5}
TP_SIZE=${TP_SIZE:-1}
SEEDS=${SEEDS:-1,2,3}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
NUM_SAMPLES=${NUM_SAMPLES:-50}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_TOKENS=${MAX_TOKENS:-256}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
GREEDY_TEMPERATURE=${GREEDY_TEMPERATURE:-0.0}
RETRY_VARIANT=${RETRY_VARIANT:-baseline}
RECOVERY_VARIANT=${RECOVERY_VARIANT:-evidence_bearing}
RUN_GROUP=${RUN_GROUP:-qwen7b_multiseed}

# Comma-separated alias=/abs/path pairs. Each path should be a directory containing JSONL files.
DATASET_ROOTS=${DATASET_ROOTS:-math=/data_x/aa007878/projects/galileo/tmp/pilot_data_grounded,nonmath=/data_x/aa007878/projects/galileo/tmp/pilot_data_arc_grounded}

CONDA_SH=${CONDA_SH:-/data_x/aa007878/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-galileo}

RESULTS_ROOT=${RESULTS_ROOT:-${REPO_DIR}/tmp/results/${RUN_GROUP}_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$RESULTS_ROOT"

RUNNER="$RESULTS_ROOT/run_multiseed.sh"
cat > "$RUNNER" <<RUNNER1
#!/usr/bin/env bash
set -euo pipefail
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$REPO_DIR"

GPU="$GPU"
TP_SIZE="$TP_SIZE"
SEEDS="$SEEDS"
MODEL="$MODEL"
NUM_SAMPLES="$NUM_SAMPLES"
MAX_MODEL_LEN="$MAX_MODEL_LEN"
MAX_TOKENS="$MAX_TOKENS"
GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION"
GREEDY_TEMPERATURE="$GREEDY_TEMPERATURE"
RETRY_VARIANT="$RETRY_VARIANT"
RECOVERY_VARIANT="$RECOVERY_VARIANT"
DATASET_ROOTS="$DATASET_ROOTS"
RESULTS_ROOT="$RESULTS_ROOT"
RUN_GROUP="$RUN_GROUP"
RUNNER1

cat >> "$RUNNER" <<'RUNNER2'
split_csv() {
  local s="$1"
  IFS=, read -r -a arr <<< "$s"
  echo "${arr[@]}"
}

declare -a dataset_pairs
IFS=, read -r -a dataset_pairs <<< "$DATASET_ROOTS"

run_one() {
  local seed="$1"
  local alias="$2"
  local data_dir="$3"
  local out_dir="${RESULTS_ROOT}/${alias}/seed_${seed}"
  mkdir -p "$out_dir"
  echo "[$(date)] seed=${seed} alias=${alias} retry=${RETRY_VARIANT} recovery=${RECOVERY_VARIANT}" | tee -a "$out_dir/run.log"

  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 \
    python /data_x/aa007878/projects/galileo/run_experiment.py \
      --model "${MODEL}" \
      --data_dir "${data_dir}" \
      --results_dir "${out_dir}" \
      --tensor_parallel_size "${TP_SIZE}" \
      --num_samples "${NUM_SAMPLES}" \
      --seed "${seed}" \
      --max_model_len "${MAX_MODEL_LEN}" \
      --max_tokens "${MAX_TOKENS}" \
      --greedy_temperature "${GREEDY_TEMPERATURE}" \
      --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
      --retry_variant "${RETRY_VARIANT}" \
      --recovery_variant "${RECOVERY_VARIANT}" \
      --personas control_reask,authority_claim \
      --enforce_eager \
      2>&1 | tee -a "$out_dir/run.log"

  python /data_x/aa007878/projects/galileo/scripts/paper_export.py \
    --results_root "${out_dir}" \
    --model_dir "${out_dir}/${MODEL##*/}" \
    --out_dir "${out_dir}/paper_exports" \
    --num_flip_samples 100 \
    --seed "${seed}" 2>&1 | tee -a "$out_dir/run.log"

  python /data_x/aa007878/projects/galileo/scripts/write_runner_metadata.py \
    --paper_exports "${out_dir}/paper_exports" \
    --model "${MODEL}" \
    --seed "${seed}" \
    --gpu_list "${GPU}" \
    --tp "${TP_SIZE}" \
    --num_samples "${NUM_SAMPLES}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --max_tokens "${MAX_TOKENS}" \
    --conda_env "${CONDA_DEFAULT_ENV}" \
    --extra_json "{\"retry_variant\":\"${RETRY_VARIANT}\",\"recovery_variant\":\"${RECOVERY_VARIANT}\",\"gpu_memory_utilization\":${GPU_MEMORY_UTILIZATION},\"dataset_alias\":\"${alias}\"}" \
    2>&1 | tee -a "$out_dir/run.log"

  python /data_x/aa007878/projects/galileo/scripts/validate_paper_exports.py \
    --results_root "${out_dir}" \
    --check_runner_parity 2>&1 | tee -a "$out_dir/run.log"
}

for seed in $(split_csv "$SEEDS"); do
  for pair in "${dataset_pairs[@]}"; do
    alias="${pair%%=*}"
    data_dir="${pair#*=}"
    run_one "$seed" "$alias" "$data_dir"
  done
done

python /data_x/aa007878/projects/galileo/scripts/validate_paper_exports.py \
  --results_root "${RESULTS_ROOT}" \
  --check_runner_parity 2>&1 | tee -a "${RESULTS_ROOT}/GLOBAL_VALIDATE.log"
RUNNER2

chmod +x "$RUNNER"

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION" -n main
fi
tmux send-keys -t "${SESSION}:main" "$RUNNER" C-m

echo "Started tmux session: $SESSION"
echo "Results root: $RESULTS_ROOT"
echo "Runner: $RUNNER"
