#!/usr/bin/env bash
set -euo pipefail

# Run fast, paper-facing preflight checks (stdlib-only).
# Intended for: before bundling / Overleaf sync / submission.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 scripts/audit_citations.py
python3 scripts/audit_acronyms.py --paper-facing \
  --acronym NRC \
  --long-form "Neutral Re-asking Control" \
  --require-first-use "Neutral Re-asking Control (NRC)"
python3 scripts/audit_paper_assets.py

echo "[OK] paper preflights passed"
