#!/usr/bin/env bash
set -euo pipefail

# Sync PDF figures into LaTeX skeleton directories (for TeX-enabled builds/Overleaf).
#
# Source of truth:
#   paper_figures/pdf/*.pdf
# Targets:
#   docs/paper/latex_skeleton/figures/
#   docs/paper/latex_skeleton_emnlp2023/figures/
#
# Usage:
#   bash scripts/sync_pdf_figures_to_latex_skeleton.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/paper_figures/pdf"

if ! compgen -G "$SRC/*.pdf" >/dev/null; then
  echo "[ERR] no PDFs found under $SRC (run SVG->PDF conversion first)" >&2
  exit 2
fi

sync_dir() {
  local dst="$1"
  mkdir -p "$dst"
  rsync -av --delete --include='*.pdf' --exclude='*' "$SRC/" "$dst/" >/dev/null
  echo "[OK] synced PDFs -> $dst" >&2
}

# Prefer rsync, fallback to cp if missing.
if ! command -v rsync >/dev/null 2>&1; then
  echo "[WARN] rsync not found; using cp (no delete)" >&2
  for dst in \
    "$ROOT/docs/paper/latex_skeleton/figures" \
    "$ROOT/docs/paper/latex_skeleton_emnlp2023/figures"; do
    mkdir -p "$dst"
    cp -a "$SRC"/*.pdf "$dst/"
    echo "[OK] copied PDFs -> $dst" >&2
  done
  exit 0
fi

sync_dir "$ROOT/docs/paper/latex_skeleton/figures"
sync_dir "$ROOT/docs/paper/latex_skeleton_emnlp2023/figures"
