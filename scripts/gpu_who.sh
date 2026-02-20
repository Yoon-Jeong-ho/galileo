#!/usr/bin/env bash
set -euo pipefail

# gpu_who.sh
# Show which PIDs are using each GPU, and map PID -> user/command.
# Intended for nlp8 where `nvidia-smi --query-compute-apps` does not expose a `username` column.
#
# Usage:
#   bash scripts/gpu_who.sh              # GPUs 0..6
#   bash scripts/gpu_who.sh 4 5 6        # explicit GPU indices

GPUS=()
if [[ $# -gt 0 ]]; then
  GPUS=("$@")
else
  GPUS=(0 1 2 3 4 5 6)
fi

for i in "${GPUS[@]}"; do
  echo "== GPU $i =="
  # Output columns: pid, process_name, used_gpu_memory [MiB]
  mapfile -t LINES < <(nvidia-smi -i "$i" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)

  if [[ ${#LINES[@]} -eq 0 ]]; then
    echo "(no compute apps)"
    echo
    continue
  fi

  printf "%8s  %-10s  %-8s  %s\n" "PID" "USER" "MiB" "CMD"
  for line in "${LINES[@]}"; do
    # line like: "12345, python, 40960"
    pid=$(echo "$line" | cut -d',' -f1 | xargs)
    mem=$(echo "$line" | cut -d',' -f3 | xargs)

    user=$(ps -o user= -p "$pid" 2>/dev/null | xargs || true)
    cmd=$(ps -o args= -p "$pid" 2>/dev/null | sed -E 's/[[:space:]]+/ /g' | cut -c1-140 || true)

    if [[ -z "$user" ]]; then user="?"; fi
    if [[ -z "$cmd" ]]; then cmd="(exited)"; fi

    printf "%8s  %-10s  %-8s  %s\n" "$pid" "$user" "$mem" "$cmd"
  done
  echo
done
