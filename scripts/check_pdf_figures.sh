#!/usr/bin/env bash
set -euo pipefail

# Smoke-check that paper figure PDFs exist and look valid.
# This does NOT compile LaTeX (LaTeX may not be installed in the runtime).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PDF_DIR="${1:-$ROOT/paper_figures/pdf}"

if [[ ! -d "$PDF_DIR" ]]; then
  echo "[ERR] missing dir: $PDF_DIR" >&2
  exit 2
fi

shopt -s nullglob
pdfs=("$PDF_DIR"/*.pdf)
if [[ ${#pdfs[@]} -eq 0 ]]; then
  echo "[ERR] no PDFs found in $PDF_DIR" >&2
  exit 3
fi

bad=0
for p in "${pdfs[@]}"; do
  size=$(wc -c <"$p" | tr -d ' ')
  head=$(head -c 5 "$p" || true)
  if [[ "$head" != "%PDF-"* ]]; then
    echo "[ERR] not a PDF (bad header): $p" >&2
    bad=1
    continue
  fi
  if [[ "$size" -lt 1024 ]]; then
    echo "[ERR] suspiciously small PDF (<1KB): $p ($size bytes)" >&2
    bad=1
    continue
  fi
  echo "[OK] $(basename "$p") ($size bytes)"
done

if [[ "$bad" -ne 0 ]]; then
  exit 4
fi

echo "[OK] PDF figure smoke-check passed: $PDF_DIR"