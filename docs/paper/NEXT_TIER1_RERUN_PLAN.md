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

### Option A — Phi‑3‑mini (4k)
- Model: `microsoft/Phi-3-mini-4k-instruct`
- Rationale: small/fast → quickest cross-family recovery coverage.

### Option B — Mistral‑Nemo
- Model: `mistralai/Mistral-Nemo-Instruct-2407`
- Rationale: stronger model family → higher external validity, but may be more expensive.

## Hard gates (fail-fast)

Run on nlp8 only.

1) **GPU availability gate** (idle + not other users)
```bash
nvidia-smi
# If a GPU looks free, map PID→user:
ps -o user= -p <pid>
```

2) **CUDA alloc preflight gate**
```bash
cd /data_x/aa007878/galileo
CUDA_VISIBLE_DEVICES=<gpu> python3 scripts/check_cuda_preflight.py
# must print: OK cuda alloc
```

3) **vLLM init preflight gate**
```bash
cd /data_x/aa007878/galileo
CUDA_VISIBLE_DEVICES=<gpu> python3 scripts/preflight_vllm_model.py --model <hf_model_id>
```

4) **Token-cap gate (post-run)**
```bash
python3 scripts/check_runlog_for_token_caps.py results/<run>/run.log
# must NOT contain "capped to 1"
```

## Canonical launch (tmux; single seed; single model)

> NOTE: use the existing canonical runner script if available; otherwise run `run_experiment.py` directly.

Template:
```bash
tmux new-session -d -s tier1-rerun \
  "set -euo pipefail; \
   cd /data_x/aa007878/galileo; \
   GPU=<gpu>; MODEL=<hf_model_id>; SEED=<1|2>; \
   OUT=results/tier1_rerun_${MODEL##*/}_seed${SEED}_$(date +%Y%m%d_%H%M%S); \
   mkdir -p $OUT; \
   echo \"GPU=$GPU MODEL=$MODEL SEED=$SEED OUT=$OUT\" | tee -a $OUT/run.log; \
   CUDA_VISIBLE_DEVICES=$GPU python3 run_experiment.py \
     --model $MODEL --seed $SEED --num_samples 1000 --tensor_parallel_size 1 \
     --max_model_len 16384 --max_tokens 2048 \
     --results_dir $OUT 2>&1 | tee -a $OUT/run.log; \
   python3 scripts/paper_export.py --results_root $OUT --model_dir $OUT/${MODEL##*/} --out_dir $OUT/paper_exports --seed $SEED --num_flip_samples 200 2>&1 | tee -a $OUT/run.log; \
   python3 scripts/write_runner_metadata.py --paper_exports $OUT/paper_exports --model $MODEL --seed $SEED --gpu_list $GPU --tp 1 --num_samples 1000 --max_model_len 16384 --max_tokens 2048 --conda_env galileo 2>&1 | tee -a $OUT/run.log; \
   python3 scripts/validate_paper_exports.py --results_root $OUT 2>&1 | tee -a $OUT/run.log; \
   python3 scripts/check_runlog_for_token_caps.py $OUT/run.log 2>&1 | tee -a $OUT/run.log; \
   echo DONE | tee -a $OUT/run.log"

# attach:
# tmux attach -t tier1-rerun
```

## After the rerun

1) Stage into `results_paper/` (via manifest + restage script).
2) Run:
```bash
python3 scripts/make_table1_from_results_paper_exports.py --results_paper results_paper --out docs/paper/artifacts/table1_from_results_paper_exports_$(date +%Y%m%d).csv
python3 scripts/gen_latex_table1_from_artifacts.py --out docs/paper/latex_paper_emnlp2023/generated/table1_rows.tex
```
3) Commit **artifacts only** (CSV under `docs/paper/artifacts/`). Do not commit full `results_paper/`.
