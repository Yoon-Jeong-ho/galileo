#!/usr/bin/env bash
set -euo pipefail

# Run fast, paper-facing preflight checks (stdlib-only).
# Intended for: before bundling / Overleaf sync / submission.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Include the LaTeX SSOT in citation audit by default.
# This catches missing BibTeX entries used in the actual submission PDF.
python3 scripts/audit_citations.py --paths \
  docs/paper/latex_paper_emnlp2023/main.tex

python3 scripts/audit_acronyms.py --paper-facing \
  --acronym NRC \
  --long-form "Neutral Re-asking Control" \
  --require-first-use "Neutral Re-asking Control (NRC)"
python3 scripts/audit_paper_assets.py

echo "[OK] paper preflights passed"
