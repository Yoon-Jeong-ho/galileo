#!/usr/bin/env bash
set -euo pipefail

# Lightweight CLI summary for run_experiment.py.
# Motivation: `python run_experiment.py --help` may import heavyweight deps (vLLM/torch)
# and can hang or be slow on shared servers.

FILE=${1:-run_experiment.py}

if [[ ! -f "$FILE" ]]; then
  echo "[ERROR] not found: $FILE" >&2
  exit 2
fi

echo "== CLI flags in $FILE (parsed from parser.add_argument lines) =="
# Extract occurrences like: parser.add_argument("--flag", ... help="...")
# This is heuristic but good enough for quick reference.
python3 - <<'PY'
import re
from pathlib import Path
p = Path("run_experiment.py")
text = p.read_text(encoding='utf-8')
pat = re.compile(r"parser\.add_argument\((.*?)\)\s*\n", re.S)
args = []
for m in pat.finditer(text):
    block = m.group(1)
    # first string literal is typically the flag
    mflag = re.search(r"\"(--[^\"]+)\"", block)
    if not mflag:
        continue
    flag = mflag.group(1)
    mhelp = re.search(r"help\s*=\s*\((.*?)\)\s*,?\s*$", block.strip(), re.S)
    if not mhelp:
        mhelp = re.search(r"help\s*=\s*\"([^\"]*)\"", block, re.S)
    help_s = None
    if mhelp:
        hs = mhelp.group(1)
        help_s = re.sub(r"\s+", " ", hs.replace("\n", " ")).strip().strip('()').strip()
    args.append((flag, help_s or ""))
for flag, help_s in args:
    print(f"{flag:24s} {help_s}")
PY
