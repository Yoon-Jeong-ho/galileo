#!/usr/bin/env python3
"""Print copy-pastable tmux commands for Tier-1 cross-family runs.

This script does NOT execute anything (safe to run locally). It exists to
standardize the run template so we don't handcraft commands under time pressure.

SSOT: Remote runs are on nlp8 (`/data_x/aa007878/galileo`) with GPUs 4/5/6 only.

Usage:
  python3 scripts/print_crossfamily_run_commands.py \
    --model_family mistral \
    --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --seeds 1 2 \
    --gpus 4 5

It prints one tmux session per (seed,gpu) pair, with:
- CUDA_VISIBLE_DEVICES=<gpu>
- OUT=results/<tag>_seed<k>_<timestamp>/

The generated command runs:
- `run_experiment.py` (writes per-run results)
- `scripts/paper_export.py` (writes `paper_exports/`)
- `scripts/validate_paper_exports.py` (sanity-check exports)

Note: you may still need to adjust `--tensor_parallel_size` for your GPU/mode.
"""

from __future__ import annotations

import argparse
import datetime as dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_family", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--gpus", nargs="+", type=int, required=True)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or f"tier1_{args.model_family}_{args.model_name}".replace("/", "-").replace(" ", "-")

    pairs = list(zip(args.seeds, (args.gpus * 100)[: len(args.seeds)]))
    print("# Tier-1 cross-family tmux launch template")
    print(f"# model_family={args.model_family}")
    print(f"# model_name={args.model_name}")
    print(f"# seeds={args.seeds}")
    print(f"# gpus={args.gpus}")
    print()

    for seed, gpu in pairs:
        session = f"{tag}_s{seed}_g{gpu}"[:120]
        out = f"results/{tag}_seed{seed}_{ts}"
        print(f"tmux new-session -d -s {session}")
        print(f"tmux send-keys -t {session} 'cd /data_x/aa007878/galileo || exit 1' C-m")
        print(f"tmux send-keys -t {session} 'mkdir -p {out}' C-m")
        print(f"tmux send-keys -t {session} 'export CUDA_VISIBLE_DEVICES={gpu}' C-m")
        print(f"tmux send-keys -t {session} 'export OUT={out}' C-m")
        print(f"tmux send-keys -t {session} 'export CONDA_BIN=/data_x/aa007878/miniconda3/bin/conda' C-m")
        print(f"tmux send-keys -t {session} 'export CONDA_ENV=galileo' C-m")
        print(f"tmux send-keys -t {session} 'export DATA_ALL_DIR=/data_x/aa007878/galileo/data_all_strict' C-m")
        print(f"tmux send-keys -t {session} 'export TP_SIZE=1' C-m")
        print()
        print("# Run experiment")
        print(f"tmux send-keys -t {session} '\\")
        print(f"CUDA_VISIBLE_DEVICES={gpu} \\")
        print(f"${{CONDA_BIN}} run -n ${{CONDA_ENV}} python run_experiment.py \\")
        print(f"  --model \"{args.model_name}\" \\")
        print(f"  --data_dir \"${{DATA_ALL_DIR}}\" \\")
        print(f"  --results_dir \"${{OUT}}\" \\")
        print(f"  --tensor_parallel_size \"${{TP_SIZE}}\" \\")
        print(f"  --num_samples 80 \\")
        print(f"  --seed {seed} \\")
        print(f"  --max_model_len 8192 \\")
        print(f"  --max_tokens 2048 \\")
        print(f"  --greedy_temperature 1.0 \\")
        print(f"  2>&1 | tee -a ${{OUT}}/run.log' C-m")
        print()
        print("# Paper exports")
        print(f"tmux send-keys -t {session} '\\")
        print(f"${{CONDA_BIN}} run -n ${{CONDA_ENV}} python scripts/paper_export.py \\")
        print(f"  --results_root \"${{OUT}}\" \\")
        print(f"  --model_dir \"${{OUT}}/{args.model_name.split('/')[-1]}\" \\")
        print(f"  --out_dir \"${{OUT}}/paper_exports\" \\")
        print(f"  --num_flip_samples 200 \\")
        print(f"  --seed {seed} \\")
        print(f"  2>&1 | tee -a ${{OUT}}/run.log' C-m")
        print()
        print("# Validate exports")
        print(f"tmux send-keys -t {session} '\\")
        print(f"${{CONDA_BIN}} run -n ${{CONDA_ENV}} python scripts/validate_paper_exports.py --results_root \"${{OUT}}\" \\")
        print(f"  2>&1 | tee -a ${{OUT}}/run.log' C-m")
        print()

if __name__ == "__main__":
    main()
