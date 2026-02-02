#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=${1:-/data_x/aa007878/galileo/data_all}
MATH_DIR=${MATH_DIR:-/data_x/aa007878/galileo/data}
QA_DIR=${QA_DIR:-/data_x/aa007878/galileo/data_qa_pilot}

mkdir -p "$OUT_DIR"

# Link all JSONL files found in both dirs (math + QA).
# This enables Plan A: one run over all datasets.

for f in "$MATH_DIR"/*.jsonl; do
  [ -e "$f" ] || continue
  ln -sf "$f" "$OUT_DIR/$(basename "$f")"
done

for f in "$QA_DIR"/*.jsonl; do
  [ -e "$f" ] || continue
  ln -sf "$f" "$OUT_DIR/$(basename "$f")"
done

echo "Unified data_dir: $OUT_DIR"
ls -1 "$OUT_DIR" | head
