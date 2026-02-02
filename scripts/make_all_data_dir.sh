#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=${1:-/data_x/aa007878/galileo/data_all}
MATH_DIR=${MATH_DIR:-/data_x/aa007878/galileo/data}
QA_DIR=${QA_DIR:-/data_x/aa007878/galileo/data_qa_pilot}

mkdir -p "$OUT_DIR"

# Link all math datasets
for f in "$MATH_DIR"/*.jsonl; do
  [ -e "$f" ] || continue
  ln -sf "$f" "$OUT_DIR/$(basename "$f")"
done

# For QA datasets, prefer 1000-line versions when present
for base in arc_easy_val squad_val triviaqa_rc_val; do
  if [ -e "$QA_DIR/${base}_1000.jsonl" ]; then
    ln -sf "$QA_DIR/${base}_1000.jsonl" "$OUT_DIR/${base}_1000.jsonl"
  fi
  # also keep the pilot files if present
  for f in "$QA_DIR/${base}_"*.jsonl; do
    [ -e "$f" ] || continue
    ln -sf "$f" "$OUT_DIR/$(basename "$f")"
  done
done

echo "Unified data_dir: $OUT_DIR"
ls -1 "$OUT_DIR" | head
