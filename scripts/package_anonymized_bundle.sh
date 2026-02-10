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
# - Includes PDF figures by default for LaTeX reliability (disable with INCLUDE_PDF=0).
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

sanitize_md() {
  # Sanitize infra-identifying strings in staged Markdown (bundle-only; does not touch sources).
  # Keep replacements simple + conservative.
  sed -E \
    -e 's#/mnt/raid6/aa007878/galileo#<REMOTE_REPO_ROOT>#g' \
    -e 's#/data_x/aa007878/galileo#<REMOTE_REPO_ROOT>#g' \
    -e 's#/mnt/raid6/#<REMOTE_PATH>/#g' \
    -e 's#/data_x/#<REMOTE_PATH>/#g' \
    -e 's#\bnlp16\b#<REMOTE_HOST>#g' \
    -e 's#\bnlp8\b#<REMOTE_HOST>#g' \
    -e 's#aa007878@[^ ]+#<REMOTE_USER>@<REMOTE_HOST>#g' \
    -e 's#\b163\.152\.[0-9]+\.[0-9]+\b#<REMOTE_IP>#g' \
    -e 's#163\.152\.\*#<REMOTE_IP>#g'
}

copy_md_sanitized() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  sanitize_md < "$src" > "$dst"
}

# --- Paper docs (paper-facing only by default) ---
# We intentionally exclude internal process docs (README/checklists/runbooks/logs)
# from the anonymized bundle unless explicitly requested.
copy_md_sanitized "$ROOT/docs/paper/PAPER_DRAFT_EN.md" "$OUT_DIR/docs/paper/PAPER_DRAFT_EN.md"
copy_md_sanitized "$ROOT/docs/paper/FIGURE_CAPTIONS.md" "$OUT_DIR/docs/paper/FIGURE_CAPTIONS.md"
# Include anonymization notes too, but sanitized (useful for the recipient).
copy_md_sanitized "$ROOT/docs/paper/ANONYMIZATION_NOTES.md" "$OUT_DIR/docs/paper/ANONYMIZATION_NOTES.md"

# --- Figures + artifacts ---
mkdir -p "$OUT_DIR/docs/paper/figures" "$OUT_DIR/docs/paper/artifacts"

# By default include only figure SVGs referenced by PAPER_DRAFT_EN.md.
# Disable SVGs entirely: INCLUDE_SVG=0
# Copy all SVGs: SVG_USED_ONLY=0
INCLUDE_SVG="${INCLUDE_SVG:-1}"
SVG_USED_ONLY="${SVG_USED_ONLY:-1}"

if [[ "$INCLUDE_SVG" == "1" ]]; then
  if [[ "$SVG_USED_ONLY" == "1" ]]; then
    used_svg=$(grep -oE "\\\\includegraphics\[[^]]*\]\{figures/[^}]+\}" "$ROOT/docs/paper/PAPER_DRAFT_EN.md" \
      | sed -E 's#.*\{figures/([^}]+)\}#\1#' \
      | sort -u)
    if [[ -z "$used_svg" ]]; then
      echo "[WARN] SVG_USED_ONLY=1 but no \\includegraphics{figures/...} refs found; copying all SVGs." >&2
      cp -a "$ROOT/docs/paper/figures"/*.svg "$OUT_DIR/docs/paper/figures/"
    else
      missing_svg=0
      while IFS= read -r base; do
        [[ -z "$base" ]] && continue
        src_svg="$ROOT/docs/paper/figures/${base}.svg"
        if [[ -f "$src_svg" ]]; then
          cp -a "$src_svg" "$OUT_DIR/docs/paper/figures/"
        else
          echo "[WARN] referenced SVG not found: $src_svg" >&2
          missing_svg=$((missing_svg+1))
        fi
      done <<< "$used_svg"
      if [[ $missing_svg -gt 0 ]]; then
        echo "[WARN] $missing_svg referenced SVGs missing; SVG bundle may be incomplete." >&2
      fi
    fi
  else
    cp -a "$ROOT/docs/paper/figures"/*.svg "$OUT_DIR/docs/paper/figures/"
  fi
fi

cp -a "$ROOT/docs/paper/artifacts"/*.csv "$OUT_DIR/docs/paper/artifacts/" || true

# Optional: include PDFs for LaTeX-friendly bundles.
# Default is ON for build reliability; disable via: INCLUDE_PDF=0 ./scripts/package_anonymized_bundle.sh
# By default we include only PDFs that are referenced by PAPER_DRAFT_EN.md (keeps bundles small).
# To include all PDFs instead: PDF_USED_ONLY=0
INCLUDE_PDF="${INCLUDE_PDF:-1}"
PDF_USED_ONLY="${PDF_USED_ONLY:-1}"

if [[ "$INCLUDE_PDF" == "1" ]]; then
  if ! compgen -G "$ROOT/paper_figures/pdf/*.pdf" >/dev/null; then
    echo "[WARN] INCLUDE_PDF=1 but no PDFs found under $ROOT/paper_figures/pdf" >&2
  else
    mkdir -p "$OUT_DIR/paper_figures/pdf"

    if [[ "$PDF_USED_ONLY" == "1" ]]; then
      # Extract figure basenames referenced in the EN draft (expects \includegraphics{figures/<name>} ).
      # We keep this heuristic simple on purpose.
      used=$(grep -oE "\\\\includegraphics\[[^]]*\]\{figures/[^}]+\}" "$ROOT/docs/paper/PAPER_DRAFT_EN.md" \
        | sed -E 's#.*\{figures/([^}]+)\}#\1#' \
        | sort -u)

      if [[ -z "$used" ]]; then
        echo "[WARN] PDF_USED_ONLY=1 but no \\includegraphics{figures/...} refs found; copying all PDFs." >&2
        cp -a "$ROOT/paper_figures/pdf"/*.pdf "$OUT_DIR/paper_figures/pdf/"
      else
        missing=0
        while IFS= read -r base; do
          [[ -z "$base" ]] && continue
          src="$ROOT/paper_figures/pdf/${base}.pdf"
          if [[ -f "$src" ]]; then
            cp -a "$src" "$OUT_DIR/paper_figures/pdf/"
          else
            echo "[WARN] referenced PDF not found: $src" >&2
            missing=$((missing+1))
          fi
        done <<< "$used"
        if [[ $missing -gt 0 ]]; then
          echo "[WARN] $missing referenced PDFs missing; LaTeX bundle may be incomplete." >&2
        fi
      fi
    else
      cp -a "$ROOT/paper_figures/pdf"/*.pdf "$OUT_DIR/paper_figures/pdf/"
    fi
  fi
fi

# --- Scripts needed to regenerate figures from artifacts ---
mkdir -p "$OUT_DIR/scripts"
copy "$ROOT/scripts/make_paper_figures_from_artifacts.py" "$OUT_DIR/scripts/make_paper_figures_from_artifacts.py"
copy "$ROOT/scripts/make_protocol_figure_svg.py" "$OUT_DIR/scripts/make_protocol_figure_svg.py"
copy "$ROOT/scripts/convert_figures_svg_to_pdf.sh" "$OUT_DIR/scripts/convert_figures_svg_to_pdf.sh"
# If PDFs are included, also include the PDF smoke-check helper.
if [[ "$INCLUDE_PDF" == "1" ]]; then
  copy "$ROOT/scripts/check_pdf_figures.sh" "$OUT_DIR/scripts/check_pdf_figures.sh"
fi

# --- Audit staged bundle for infra-identifying strings ---
PAT='(/mnt/raid6/|/data_x/|nlp16|nlp8|aa007878@|163\.152\.|ssh nlp)'
if grep -RIn --exclude-dir='__pycache__' --exclude='*.svg' --exclude='*.png' --exclude='*.pdf' -E "$PAT" "$OUT_DIR"; then
  echo "[ERR] infra-identifying strings found in staged bundle. Fix before packaging." >&2
  exit 2
fi

# Best-effort PDF string scan (PDFs are binary; some generators embed metadata).
if [[ "$INCLUDE_PDF" == "1" ]] && compgen -G "$OUT_DIR/paper_figures/pdf/*.pdf" >/dev/null; then
  if command -v strings >/dev/null 2>&1; then
    if strings "$OUT_DIR"/paper_figures/pdf/*.pdf | grep -E "$PAT" >/dev/null; then
      echo "[ERR] infra-identifying strings found in PDF contents/metadata (via strings)." >&2
      exit 3
    fi
  else
    echo "[WARN] 'strings' not available; skipping best-effort PDF metadata scan." >&2
  fi
fi

echo "[OK] staged anonymized bundle at: $OUT_DIR"