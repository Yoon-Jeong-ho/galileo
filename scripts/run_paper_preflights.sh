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

# Fail fast on common drafting markers that can slip into submission.
# (Keep this stdlib-only and grep-based.)
MARKER_RE='(^|[^A-Za-z])(TODO|TBD|FIXME|XXX)([^A-Za-z]|$)'
MARKER_PATHS=(
  docs/paper/PAPER_DRAFT_EN.md
  docs/paper/ABSTRACT_EN.md
  docs/paper/FIGURE_CAPTIONS.md
  docs/paper/latex_paper_emnlp2023/main.tex
)

if grep -RInE "$MARKER_RE" "${MARKER_PATHS[@]}" >/tmp/paper_preflight_markers.txt; then
  echo "[FAIL] Found drafting markers (TODO/TBD/FIXME/XXX) in paper-facing files:" >&2
  cat /tmp/paper_preflight_markers.txt >&2
  exit 1
fi


# Structural LaTeX guardrails (prevent accidental deletion during refactors)
LATEX_MAIN="docs/paper/latex_paper_emnlp2023/main.tex"

if ! grep -nE '^\\bibliographystyle\{acl_natbib\}' "$LATEX_MAIN" >/dev/null; then
  echo "[FAIL] Missing \\bibliographystyle{acl_natbib} in $LATEX_MAIN" >&2
  exit 1
fi
if ! grep -nE '^\\bibliography\{' "$LATEX_MAIN" >/dev/null; then
  echo "[FAIL] Missing \\bibliography{...} in $LATEX_MAIN" >&2
  exit 1
fi
if ! grep -nE '^\\section\{Limitations\}' "$LATEX_MAIN" >/dev/null; then
  echo "[FAIL] Missing \\section{Limitations} in $LATEX_MAIN" >&2
  exit 1
fi
if ! grep -nE '^\\section\{Ethics Statement\}' "$LATEX_MAIN" >/dev/null; then
  echo "[FAIL] Missing \\section{Ethics Statement} in $LATEX_MAIN" >&2
  exit 1
fi

echo "[OK] paper preflights passed"
