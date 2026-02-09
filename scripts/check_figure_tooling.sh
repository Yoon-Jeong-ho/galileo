#!/usr/bin/env bash
set -euo pipefail

# Quick preflight for the LaTeX figure pipeline.
# Checks whether SVG->PDF conversion tooling is available.

ok=true

if command -v rsvg-convert >/dev/null 2>&1; then
  echo "[OK] rsvg-convert found: $(command -v rsvg-convert)"
else
  echo "[WARN] rsvg-convert not found (recommended)."
  ok=false
fi

if command -v inkscape >/dev/null 2>&1; then
  echo "[OK] inkscape found: $(command -v inkscape)"
else
  echo "[INFO] inkscape not found (optional alternative)."
fi

if [[ "$ok" != true ]]; then
  cat <<'EOF'

To enable SVG→PDF conversion for LaTeX builds, install ONE of:

  Ubuntu/Debian (recommended):
    sudo apt-get update && sudo apt-get install -y librsvg2-bin

  Alternative:
    sudo apt-get update && sudo apt-get install -y inkscape

Then run:
  ./scripts/convert_figures_svg_to_pdf.sh
EOF
  exit 2
fi

echo "[OK] Figure tooling looks good."