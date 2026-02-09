#!/usr/bin/env python3
"""Print copy-pastable tmux commands for Tier-1 cross-family runs.

This script does NOT execute anything (safe to run locally). It exists to
standardize the run template so we don't handcraft commands under time pressure.

Usage:
  python3 scripts/print_crossfamily_run_commands.py \
    --model_family mistral \
    --model_name Mistral-7B-Instruct-v0.3 \
    --seeds 1 2 \
    --gpus 4 5

It prints one tmux session per (seed,gpu) pair, with:
- CUDA_VISIBLE_DEVICES=<gpu>
- OUT=results/<tag>_seed<k>_<timestamp>/

You still need to fill in the actual runner invocation once we confirm the
canonical entrypoint + args in the remote repo.
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
        print(f"tmux send-keys -t {session} 'cd /mnt/raid6/aa007878/galileo' C-m")
        print(f"tmux send-keys -t {session} 'export CUDA_VISIBLE_DEVICES={gpu}' C-m")
        print(f"tmux send-keys -t {session} 'export OUT={out}' C-m")
        print("# TODO: replace the next line with the canonical runner command once confirmed")
        print(f"tmux send-keys -t {session} 'echo "
              f"\"[TEMPLATE] seed={seed} gpu={gpu} model={args.model_name} OUT=$OUT\""
              f" | tee -a $OUT/run.log' C-m")
        print("# e.g., python run_experiment.py --model ... --seed ... --out $OUT ...")
        print()


if __name__ == "__main__":
    main()
