#!/usr/bin/env bash
set -euo pipefail

# Quick per-GPU CUDA alloc preflight table for nlp8.
# Usage:
#   bash scripts/check_cuda_preflight_all.sh            # GPUs 0..6
#   bash scripts/check_cuda_preflight_all.sh 1 2 3 4    # explicit list
#
# This wraps scripts/check_cuda_preflight.py and treats its exit code as truth.
# If a GPU fails even though nvidia-smi looks idle, exclude it for that heartbeat.

if [[ $# -gt 0 ]]; then
  GPUS=("$@")
else
  GPUS=(0 1 2 3 4 5 6)
fi

printf "%-6s %-10s\n" "GPU" "PREFLIGHT"
for g in "${GPUS[@]}"; do
  if CUDA_VISIBLE_DEVICES="$g" python3 scripts/check_cuda_preflight.py >/dev/null 2>&1; then
    printf "%-6s %-10s\n" "$g" "OK"
  else
    printf "%-6s %-10s\n" "$g" "FAIL"
  fi
done
