#!/usr/bin/env bash
set -euo pipefail

# Build the working GALILEO EMNLP paper PDF.
#
# This is a lightweight alternative to the Makefile targets (some environments
# inside OpenClaw/WSL containers may not ship with `make`).
#
# Usage:
#   bash scripts/build_paper.sh            # review-mode build (line numbers)
#   bash scripts/build_paper.sh --camera-ready
#   bash scripts/build_paper.sh --clean    # clean latexmk artifacts
#
# Output:
#   docs/paper/latex_paper_emnlp2023/main.pdf

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PAPER_DIR="docs/paper/latex_paper_emnlp2023"
MAIN_TEX="$PAPER_DIR/main.tex"

if [[ ! -f "$MAIN_TEX" ]]; then
  echo "[ERROR] Missing $MAIN_TEX" >&2
  exit 2
fi

CAMERA_READY=0
CLEAN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --camera-ready)
      CAMERA_READY=1
      shift
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      echo "Run: bash scripts/build_paper.sh --help" >&2
      exit 2
      ;;
  esac
done

if [[ "$CLEAN" -eq 1 ]]; then
  echo "[INFO] Cleaning LaTeX artifacts under $PAPER_DIR" >&2
  (cd "$PAPER_DIR" && latexmk -C)
  exit 0
fi

# Fail fast if citation keys are missing.
./scripts/check_citations_vs_bib.sh

echo "[INFO] Building paper PDF under $PAPER_DIR" >&2
if [[ "$CAMERA_READY" -eq 1 ]]; then
  (cd "$PAPER_DIR" && latexmk -pdf -interaction=nonstopmode -halt-on-error \
    -pdflatex='pdflatex %O "\\def\\CAMERAREADY{1}\\input{%S}"' main.tex)
else
  (cd "$PAPER_DIR" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
fi

echo "[OK] Built: $PAPER_DIR/main.pdf" >&2
