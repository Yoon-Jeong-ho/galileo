#!/usr/bin/env bash
set -euo pipefail

SESSION=${1:-galileo-multiseed-families}

GPU_LIST=${GPU_LIST:-4,5,6,7}
TP_SIZE=${TP_SIZE:-4}

DATA_ALL_DIR=${DATA_ALL_DIR:-/data_x/aa007878/galileo/data_all_strict}
MATH_DIR=${MATH_DIR:-/data_x/aa007878/galileo/data}
QA_DIR=${QA_DIR:-/data_x/aa007878/galileo/data_qa_full}

RESULTS_ROOT=${RESULTS_ROOT:-/mnt/raid6/aa007878/galileo/results/multiseed_families_$(date +%Y%m%d_%H%M%S)}
NUM_SAMPLES=${NUM_SAMPLES:-1000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_TOKENS=${MAX_TOKENS:-2048}
SEEDS=${SEEDS:-1,2,3,4,5}

# Temperature sweep for adversarial/recovery decoding
GREEDY_TEMPS=${GREEDY_TEMPS:-1.0,0.7}

CONDA_BIN=${CONDA_BIN:-/data_x/aa007878/miniconda3/bin/conda}
CONDA_ENV=${CONDA_ENV:-galileo}

# tag=model_id pairs (comma-separated)
MODELS=${MODELS:-llama3_8b=meta-llama/Meta-Llama-3-8B-Instruct,llama3_3b=meta-llama/Llama-3.2-3B-Instruct,mistral7b=mistralai/Mistral-7B-Instruct-v0.3,mistral12b=mistralai/Mistral-Nemo-Instruct-2407,exaone7_8b=LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct}

mkdir -p "$RESULTS_ROOT"

# Build strict unified data dir
MATH_DIR="$MATH_DIR" QA_DIR="$QA_DIR" bash scripts/make_all_data_dir_strict.sh "$DATA_ALL_DIR" >/dev/null

RUNNER="$RESULTS_ROOT/run_multiseed_families.sh"

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
GREEDY_TEMPS="$GREEDY_TEMPS"
CONDA_BIN="$CONDA_BIN"
CONDA_ENV="$CONDA_ENV"
MODELS="$MODELS"
RUN1

cat >> "$RUNNER" <<'RUN2'

echo "=== Galileo multiseed families start: $(date) ==="
echo "GPUs: ${GPU_LIST} / TP: ${TP_SIZE}"
echo "RESULTS_ROOT: ${RESULTS_ROOT}"
echo "MODELS: ${MODELS}"
echo "SEEDS: ${SEEDS}"
echo "GREEDY_TEMPS: ${GREEDY_TEMPS}"

to_array() {
  local s="$1"
  IFS=, read -r -a arr <<< "$s"
  echo "${arr[@]}"
}

# Preflight model existence (skip on 404/auth errors)
preflight() {
  local model="$1"
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python - "$model" <<'PY'
from huggingface_hub import model_info
import sys
m=sys.argv[1]
try:
    model_info(m)
    print('OK')
except Exception as e:
    print('FAIL', type(e).__name__, str(e)[:200])
    sys.exit(2)
PY
}

run_one() {
  local seed="$1"
  local tag="$2"
  local model="$3"
  local temp="$4"

  local out_dir="${RESULTS_ROOT}/seed_${seed}/${tag}/temp_${temp}"
  mkdir -p "$out_dir"

  echo "[$(date)] seed=${seed} tag=${tag} model=${model} temp=${temp}" | tee -a "$out_dir/run.log"

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
    --greedy_temperature "${temp}" \
    2>&1 | tee -a "$out_dir/run.log"

  "${CONDA_BIN}" run -n "${CONDA_ENV}" python scripts/paper_export.py \
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
  "greedy_temperature": ${temp},
  "conda_env": "${CONDA_ENV}",
  "model": "${model}",
  "seed": ${seed},
  "tag": "${tag}"
}
JSON
}

seeds=( $(to_array "${SEEDS}") )
temps=( $(to_array "${GREEDY_TEMPS}") )

# parse MODELS
declare -a tags
declare -a model_ids
IFS=, read -r -a pairs <<< "${MODELS}"
for pair in "${pairs[@]}"; do
  tag="${pair%%=*}"
  model="${pair#*=}"
  tags+=("${tag}")
  model_ids+=("${model}")
done

for idx in "${!tags[@]}"; do
  tag="${tags[$idx]}"
  model="${model_ids[$idx]}"
  echo "--- preflight ${tag}=${model} ---"
  if ! preflight "${model}"; then
    echo "SKIP: ${model} (preflight failed)" | tee -a "${RESULTS_ROOT}/SKIPPED_MODELS.log"
    continue
  fi

  for seed in "${seeds[@]}"; do
    for temp in "${temps[@]}"; do
      run_one "${seed}" "${tag}" "${model}" "${temp}"
    done
  done
done

echo "=== Galileo multiseed families done: $(date) ==="
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
