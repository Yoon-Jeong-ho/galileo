#!/usr/bin/env bash
set -euo pipefail

# Run fast, paper-facing preflight checks (stdlib-only).
# Intended for: before bundling / Overleaf sync / submission.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Optional: include LaTeX SSOT files in citation audit.
# Usage:
#   PREFLIGHT_LATEX_SSOT=1 bash scripts/run_paper_preflights.sh
PREFLIGHT_LATEX_SSOT="${PREFLIGHT_LATEX_SSOT:-0}"

if [[ "$PREFLIGHT_LATEX_SSOT" == "1" ]]; then
  python3 scripts/audit_citations.py --paths \
    docs/paper/latex_paper_emnlp2023/main.tex \
    docs/paper/PAPER_DRAFT_EN.md \
    docs/paper/PAPER_DRAFT_KO.md
else
  python3 scripts/audit_citations.py
fi

python3 scripts/audit_acronyms.py --paper-facing \
  --acronym NRC \
  --long-form "Neutral Re-asking Control" \
  --require-first-use "Neutral Re-asking Control (NRC)"
python3 scripts/audit_paper_assets.py

echo "[OK] paper preflights passed"
