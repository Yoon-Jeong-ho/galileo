# Next Tier‑1 rerun plan (SSOT: nlp8)

Last updated: 2026-02-24

## Why a rerun is needed

As of 2026-02-24, on nlp8 SSOT (`/data_x/aa007878/galileo`):
- `results/` (maxdepth=1) contains Phi‑3.5‑mini runs, but **no** `*phi3mini*` or `*mistralnemo*` run dirs.
- `results_paper/` currently has no `phi|nemo` aliases staged.

So we cannot “backfill recovery exports” for Phi‑3‑mini / Mistral‑Nemo without either:
- recovering archived run directories, or
- rerunning seed1–2 under standardized Tier‑1 settings.

This note defines a **single-run** plan (seed1 or seed2) that is reviewer-risk-optimal.

## Candidate targets (pick ONE)

### Recommended default (unless you explicitly want Nemo)
**Phi‑3‑mini seed1** first. It’s the fastest way to restore “missing-family Recovery@flip coverage” in Table 1 with minimal GPU risk.

### Option A — Phi‑3‑mini (4k)
- Model: `microsoft/Phi-3-mini-4k-instruct`
- Rationale: small/fast → quickest cross-family recovery coverage.

### Option B — Mistral‑Nemo
- Model: `mistralai/Mistral-Nemo-Instruct-2407`
- Rationale: stronger model family → higher external validity, but may be more expensive.

## Hard gates (fail-fast)

Run on nlp8 only.

> **Important:** on nlp8, system `python3` may not have torch/vLLM. Use the conda env:
> `CONDA_RUN="/data_x/aa007878/miniconda3/bin/conda run -n galileo"`

1) **GPU availability gate** (idle + not other users)
```bash
nvidia-smi
# If a GPU looks free, map PID→user:
ps -o user= -p <pid>
```

2) **CUDA alloc preflight gate**
```bash
cd /data_x/aa007878/galileo
CONDA_RUN="/data_x/aa007878/miniconda3/bin/conda run -n galileo"
CUDA_VISIBLE_DEVICES=<gpu> $CONDA_RUN python3 scripts/check_cuda_preflight.py
# must print: OK cuda alloc
```

3) **vLLM init preflight gate**
```bash
cd /data_x/aa007878/galileo
CONDA_RUN="/data_x/aa007878/miniconda3/bin/conda run -n galileo"
CUDA_VISIBLE_DEVICES=<gpu> $CONDA_RUN python3 scripts/preflight_vllm_model.py --model <hf_model_id>
```

4) **Token-cap gate (post-run)**
```bash
cd /data_x/aa007878/galileo
CONDA_RUN="/data_x/aa007878/miniconda3/bin/conda run -n galileo"
$CONDA_RUN python3 scripts/check_runlog_for_token_caps.py results/<run>/run.log
# must NOT contain "capped to 1"
```

## Canonical launch (tmux; single seed; single model)

> NOTE: use the existing canonical runner script if available; otherwise run `run_experiment.py` directly.

Template (Tier‑1 = **6-benchmark SSOT**; conda required):
```bash
tmux new-session -d -s tier1-rerun \
  "set -euo pipefail; \
   cd /data_x/aa007878/galileo; \
   CONDA_RUN='/data_x/aa007878/miniconda3/bin/conda run -n galileo'; \
   GPU=<gpu>; MODEL=<hf_model_id>; SEED=<1|2>; \
   DATA_DIR=/data_x/aa007878/galileo/data_tier1_6; \
   OUT=results/tier1_rerun_${MODEL##*/}_seed${SEED}_$(date +%Y%m%d_%H%M%S); \
   mkdir -p $OUT; \
   echo \"GPU=$GPU MODEL=$MODEL SEED=$SEED DATA_DIR=$DATA_DIR OUT=$OUT\" | tee -a $OUT/run.log; \
   # NOTE: stdout can be block-buffered; force line-buffering so run.log updates live.
   CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 stdbuf -oL -eL $CONDA_RUN python3 -u run_experiment.py \
     --model $MODEL --seed $SEED --num_samples 1000 --tensor_parallel_size 1 \
     --data_dir $DATA_DIR \
     --max_model_len 4096 --max_tokens 2048 \
     --results_dir $OUT 2>&1 | tee -a $OUT/run.log; \
   $CONDA_RUN python3 scripts/paper_export.py --results_root $OUT --model_dir $OUT/${MODEL##*/} --out_dir $OUT/paper_exports --seed $SEED --num_flip_samples 200 2>&1 | tee -a $OUT/run.log; \
   $CONDA_RUN python3 scripts/write_runner_metadata.py --paper_exports $OUT/paper_exports --model $MODEL --seed $SEED --gpu_list $GPU --tp 1 --num_samples 1000 --max_model_len 4096 --max_tokens 2048 --conda_env galileo 2>&1 | tee -a $OUT/run.log; \
   $CONDA_RUN python3 scripts/validate_paper_exports.py --results_root $OUT 2>&1 | tee -a $OUT/run.log; \
   $CONDA_RUN python3 scripts/check_runlog_for_token_caps.py $OUT/run.log 2>&1 | tee -a $OUT/run.log; \
   echo DONE | tee -a $OUT/run.log"

# attach:
# tmux attach -t tier1-rerun
```

Notes:
- Prefer a **pilot first** (`--num_samples 200 --max_tokens 512`) if the model/stack is flaky; only scale up after validator + token-cap gate.
- If `run.log` looks silent but GPU is busy, check output growth under `$OUT/<model_name>/*.jsonl` (size/mtime) to confirm progress.
- **Stall watchdog (recommended):** if *no* JSONL file under `$OUT/<model_name>/` changes mtime for **>10 minutes**, treat as a hang and kill the tmux session to free the GPU.
  - Quick check:
    ```bash
    OUT=results/<run>
    find $OUT -name '*.jsonl' -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort | tail -n 5
    ```
- `--max_model_len` must respect the model’s true context window (Phi‑3‑mini‑4k → 4096).

## After the rerun

1) Stage into `results_paper/` (via manifest + restage script).
2) Run:
```bash
python3 scripts/make_table1_from_results_paper_exports.py --results_paper results_paper --out docs/paper/artifacts/table1_from_results_paper_exports_$(date +%Y%m%d).csv
python3 scripts/gen_latex_table1_from_artifacts.py --out docs/paper/latex_paper_emnlp2023/generated/table1_rows.tex
```
3) Commit **artifacts only** (CSV under `docs/paper/artifacts/`). Do not commit full `results_paper/`.
