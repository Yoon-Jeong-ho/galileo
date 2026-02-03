#!/usr/bin/env bash
set -euo pipefail

# Create a unified data dir that links ONLY the intended "main" datasets.
# This avoids accidentally including old pilot files like *_val_50.jsonl.

OUT_DIR=${1:-/data_x/aa007878/galileo/data_all_strict}
MATH_DIR=${MATH_DIR:-/data_x/aa007878/galileo/data}
QA_DIR=${QA_DIR:-/data_x/aa007878/galileo/data_qa_full}

mkdir -p "$OUT_DIR"

link_one() {
  local src="$1"
  local dst_name="$2"
  if [ ! -e "$src" ]; then
    echo "[missing] $src" >&2
    return 1
  fi
  ln -sf "$src" "$OUT_DIR/$dst_name"
}

# Math (full)
link_one "$MATH_DIR/gsm8k.jsonl" "gsm8k.jsonl"
link_one "$MATH_DIR/svamp.jsonl" "svamp.jsonl"

# QA/MCQA (full)
link_one "$QA_DIR/squad11_validation.jsonl" "squad11_validation.jsonl"
link_one "$QA_DIR/squad20_validation.jsonl" "squad20_validation.jsonl"
link_one "$QA_DIR/arc_easy_validation.jsonl" "arc_easy_validation.jsonl"
link_one "$QA_DIR/triviaqa_rc_validation.jsonl" "triviaqa_rc_validation.jsonl"

# sanity: list
echo "Unified (strict) data_dir: $OUT_DIR"
for f in "$OUT_DIR"/*.jsonl; do
  echo -n "$(basename "$f") "
  wc -l "$f"
done
