#!/usr/bin/env bash
set -euo pipefail

# Convert submission figures from SVG -> PDF.
# Default input: docs/paper/figures/*.svg
# Default output: paper_figures/pdf/*.pdf
#
# Rationale:
# - Many LaTeX/Overleaf pipelines are happiest with PDF/PNG.
# - We keep SVGs as the source of truth and regenerate PDFs.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IN_DIR="${1:-$ROOT/docs/paper/figures}"
OUT_DIR="${2:-$ROOT/paper_figures/pdf}"

mkdir -p "$OUT_DIR"

INKSCAPE_APPIMAGE_DEFAULT="$ROOT/tools/inkscape/inkscape.AppImage"
INKSCAPE_APPIMAGE="${INKSCAPE_APPIMAGE:-$INKSCAPE_APPIMAGE_DEFAULT}"

if command -v rsvg-convert >/dev/null 2>&1; then
  CONVERTER="rsvg"
elif command -v inkscape >/dev/null 2>&1; then
  CONVERTER="inkscape"
elif [[ -x "$INKSCAPE_APPIMAGE" ]]; then
  # No sudo: use a user-local Inkscape AppImage.
  CONVERTER="inkscape_appimage"
else
  echo "[ERR] Need rsvg-convert (librsvg) or inkscape to convert SVG->PDF." >&2
  echo "      If you have sudo: sudo apt-get install -y librsvg2-bin   (provides rsvg-convert)" >&2
  echo "      or:          sudo apt-get install -y inkscape" >&2
  echo "      If you do NOT have sudo: download an Inkscape AppImage:" >&2
  echo "        bash scripts/get_inkscape_appimage.sh" >&2
  echo "      (expects to create: $INKSCAPE_APPIMAGE_DEFAULT)" >&2
  exit 2
fi

shopt -s nullglob
svgs=("$IN_DIR"/*.svg)
if [[ ${#svgs[@]} -eq 0 ]]; then
  echo "[ERR] No .svg files found in $IN_DIR" >&2
  exit 3
fi

for svg in "${svgs[@]}"; do
  base="$(basename "$svg" .svg)"
  out="$OUT_DIR/$base.pdf"
  if [[ "$CONVERTER" == "rsvg" ]]; then
    rsvg-convert -f pdf -o "$out" "$svg"
  elif [[ "$CONVERTER" == "inkscape" ]]; then
    # Inkscape 1.x CLI
    inkscape "$svg" --export-type=pdf --export-filename="$out" >/dev/null
  else
    # AppImage CLI is identical to inkscape.
    "$INKSCAPE_APPIMAGE" "$svg" --export-type=pdf --export-filename="$out" >/dev/null
  fi
  echo "[OK] $svg -> $out"
done

echo "[OK] wrote PDFs to $OUT_DIR"