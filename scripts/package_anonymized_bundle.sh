#!/usr/bin/env bash
set -euo pipefail

# Create a minimal anonymized paper bundle (docs + figures + artifacts + figure scripts).
# This does NOT modify sources; it stages a bundle directory you can zip/tar.
#
# Usage:
#   ./scripts/package_anonymized_bundle.sh [OUT_DIR]
#
# Notes:
# - Excludes internal-only process docs (heartbeat logs/runbooks/KO notes) by default.
# - Runs an infra-string grep audit on the staged bundle and fails if matches are found.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT/tmp/anonymized_bundle}"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

copy() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  cp -a "$src" "$dst"
}

# --- Paper drafts (EN only by default) ---
copy "$ROOT/docs/paper/PAPER_DRAFT_EN.md" "$OUT_DIR/docs/paper/PAPER_DRAFT_EN.md"
copy "$ROOT/docs/paper/FIGURE_CAPTIONS.md" "$OUT_DIR/docs/paper/FIGURE_CAPTIONS.md"
# Intentionally NOT copying ANONYMIZATION_NOTES.md into the anonymized bundle,
# because it necessarily contains infra-identifying strings (it is the audit SSOT).

# --- Figures + artifacts ---
mkdir -p "$OUT_DIR/docs/paper/figures" "$OUT_DIR/docs/paper/artifacts"
cp -a "$ROOT/docs/paper/figures"/*.svg "$OUT_DIR/docs/paper/figures/"
cp -a "$ROOT/docs/paper/artifacts"/*.csv "$OUT_DIR/docs/paper/artifacts/" || true

# --- Scripts needed to regenerate figures from artifacts ---
mkdir -p "$OUT_DIR/scripts"
copy "$ROOT/scripts/make_paper_figures_from_artifacts.py" "$OUT_DIR/scripts/make_paper_figures_from_artifacts.py"
copy "$ROOT/scripts/make_protocol_figure_svg.py" "$OUT_DIR/scripts/make_protocol_figure_svg.py"
copy "$ROOT/scripts/convert_figures_svg_to_pdf.sh" "$OUT_DIR/scripts/convert_figures_svg_to_pdf.sh"

# --- Audit staged bundle for infra-identifying strings ---
PAT='(/mnt/raid6/|/data_x/|nlp16|nlp8|aa007878@|163\.152\.|ssh nlp)'
if grep -RIn --exclude-dir='__pycache__' --exclude='*.svg' --exclude='*.png' -E "$PAT" "$OUT_DIR"; then
  echo "[ERR] infra-identifying strings found in staged bundle. Fix before packaging." >&2
  exit 2
fi

echo "[OK] staged anonymized bundle at: $OUT_DIR"