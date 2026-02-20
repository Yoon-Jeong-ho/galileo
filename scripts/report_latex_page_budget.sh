#!/usr/bin/env bash
set -euo pipefail

# Report page budgeting for the LaTeX SSOT (PDF-first workflow).
# Computes:
#   - total pages
#   - main pages (everything before \appendix)
#   - main pages excluding the "Discussion and limitations" section
#
# This relies on \pagemark{...} markers emitted into the LaTeX log.

TEX_DIR="docs/paper/latex_paper_emnlp2023"
TEX_FILE="main.tex"

cd "$(git rev-parse --show-toplevel)"
cd "$TEX_DIR"

# Build camera-ready-ish (no [review] line numbers) to approximate submission page count.
latexmk -C >/dev/null 2>&1 || true
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -pdflatex='pdflatex %O "\\def\\CAMERAREADY{1}\\input{%S}"' \
  "$TEX_FILE" >/dev/null

TOTAL_PAGES=$(pdfinfo main.pdf | awk -F: '/^Pages:/ {gsub(/ /, "", $2); print $2}')

# Extract pagemarks from the log.
LIMIT_START=$(awk -F= '/^PAGE_MARK:LIMITATIONS_SECTION_START=/ {print $2; exit}' main.log || true)
APPENDIX_START=$(awk -F= '/^PAGE_MARK:APPENDIX_START=/ {print $2; exit}' main.log || true)

if [[ -z "${LIMIT_START:-}" || -z "${APPENDIX_START:-}" ]]; then
  echo "[ERROR] Missing PAGE_MARK entries in main.log. Recompile and ensure main.tex defines and calls \\pagemark." >&2
  exit 2
fi

MAIN_PAGES=$((APPENDIX_START - 1))
MAIN_EXCL_LIMIT_PAGES=$((LIMIT_START - 1))

cat <<EOF
LaTeX page budget (camera-ready build; ${TEX_DIR}/${TEX_FILE})
- total_pages: ${TOTAL_PAGES}
- main_pages (before appendix): ${MAIN_PAGES}
- main_pages_excluding_limitations: ${MAIN_EXCL_LIMIT_PAGES}
- markers:
  - limitations_section_start_page: ${LIMIT_START}
  - appendix_start_page: ${APPENDIX_START}
EOF
