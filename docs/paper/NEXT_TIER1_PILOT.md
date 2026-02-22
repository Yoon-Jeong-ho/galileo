# Next Tier‑1 pilot (seed1, 50 samples)

Goal: run a **fail-fast** cross-family pilot (1 seed, 50 samples) on nlp8 using **GPU 4/5/6** with CUDA alloc preflight + token-cap gating.

## Preflight (required)

On nlp8:

```bash
cd /data_x/aa007878/galileo || exit 1
nvidia-smi
bash scripts/check_cuda_preflight_all.sh
```

Pick a GPU in {4,5,6} that is **idle** and shows **OK** in preflight.

## Generate tmux launch commands (recommended)

Locally (or on nlp8), print the standardized commands:

```bash
python3 scripts/print_crossfamily_run_commands.py \
  --model_family <short_tag> \
  --model_name <hf_id> \
  --seeds 1 \
  --gpus <GPU_ID> \
  --num_samples 50 \
  --max_model_len 4096 \
  --max_tokens 2048 \
  --greedy_temperature 1.0
```

Then paste the printed tmux commands on nlp8.

## Fail-fast cap check

While running, abort the pilot if `run.log` shows repeated max-token caps (e.g., capped to 1).

```bash
python3 scripts/check_runlog_for_token_caps.py results/<run>/run.log
```

If the pilot passes (exports + validator OK), proceed to Tier‑1 seeds 1–2 (80 samples/seed).
