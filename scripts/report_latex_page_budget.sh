#!/usr/bin/env bash
set -euo pipefail

# Report page budgeting for the LaTeX SSOT (PDF-first workflow).
# Computes:
#   - total pages
#   - main pages (everything before \appendix)
#   - main pages excluding Limitations (if the marker is present)
#   - main pages excluding Ethics (if the marker is present)
#
# This relies on \pagemark{...} markers emitted into the LaTeX log.

TEX_DIR="docs/paper/latex_paper_emnlp2023"
TEX_FILE="main.tex"

cd "$(git rev-parse --show-toplevel)"

# Dependency checks (fail fast with actionable messages)
need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[ERROR] Missing required command: $1" >&2
    case "$1" in
      pdfinfo)
        echo "        Install poppler-utils (e.g., 'sudo apt-get install -y poppler-utils')." >&2
        ;;
      latexmk)
        echo "        Install latexmk + a LaTeX distribution (e.g., TeX Live)." >&2
        ;;
      python3)
        echo "        Install Python 3." >&2
        ;;
    esac
    exit 127
  fi
}

need_cmd python3
need_cmd latexmk
need_cmd pdfinfo

cd "$TEX_DIR"

# Ensure generated LaTeX fragments (e.g., Table 1 rows) exist before compiling.
# (These are gitignored by design.)
REPO_ROOT="$(git rev-parse --show-toplevel)"
python3 "$REPO_ROOT/scripts/gen_latex_table1_from_artifacts.py" \
  --out "$REPO_ROOT/$TEX_DIR/generated/table1_rows.tex" >/dev/null

# Build camera-ready-ish (no [review] line numbers) to approximate submission page count.
latexmk -C >/dev/null 2>&1 || true
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -pdflatex='pdflatex %O "\\def\\CAMERAREADY{1}\\input{%S}"' \
  "$TEX_FILE" >/dev/null

TOTAL_PAGES=$(pdfinfo main.pdf | awk -F: '/^Pages:/ {gsub(/ /, "", $2); print $2}')

# Extract pagemarks from the log.
LIMIT_START=$(awk -F= '/^PAGE_MARK:LIMITATIONS_SECTION_START=/ {print $2; exit}' main.log || true)
ETHICS_START=$(awk -F= '/^PAGE_MARK:ETHICS_SECTION_START=/ {print $2; exit}' main.log || true)
APPENDIX_START=$(awk -F= '/^PAGE_MARK:APPENDIX_START=/ {print $2; exit}' main.log || true)

if [[ -z "${APPENDIX_START:-}" ]]; then
  echo "[ERROR] Missing PAGE_MARK:APPENDIX_START in main.log. Recompile and ensure main.tex defines and calls \\pagemark." >&2
  exit 2
fi

MAIN_PAGES=$((APPENDIX_START - 1))

MAIN_EXCL_LIMIT_PAGES=""
if [[ -n "${LIMIT_START:-}" ]]; then
  MAIN_EXCL_LIMIT_PAGES=$((LIMIT_START - 1))
fi

MAIN_EXCL_ETHICS_PAGES=""
if [[ -n "${ETHICS_START:-}" ]]; then
  MAIN_EXCL_ETHICS_PAGES=$((ETHICS_START - 1))
fi

cat <<EOF
LaTeX page budget (camera-ready build; ${TEX_DIR}/${TEX_FILE})
- total_pages: ${TOTAL_PAGES}
- main_pages (before appendix): ${MAIN_PAGES}
EOF

if [[ -n "${MAIN_EXCL_LIMIT_PAGES}" ]]; then
  echo "- main_pages_excluding_limitations: ${MAIN_EXCL_LIMIT_PAGES}"
else
  echo "- main_pages_excluding_limitations: (missing marker: LIMITATIONS_SECTION_START)"
fi

if [[ -n "${MAIN_EXCL_ETHICS_PAGES}" ]]; then
  echo "- main_pages_excluding_ethics: ${MAIN_EXCL_ETHICS_PAGES}"
else
  echo "- main_pages_excluding_ethics: (missing marker: ETHICS_SECTION_START)"
fi

cat <<EOF
- markers:
  - limitations_section_start_page: ${LIMIT_START:-}
  - ethics_section_start_page: ${ETHICS_START:-}
  - appendix_start_page: ${APPENDIX_START}
EOF
