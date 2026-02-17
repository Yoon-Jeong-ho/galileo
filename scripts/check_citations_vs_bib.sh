#!/usr/bin/env bash
set -euo pipefail

# Check that all \cite{...} keys referenced in the main paper draft exist in references.bib.
# This is meant to be a lightweight, dependency-free guardrail (no Python required).

DRAFT_PATH=${1:-docs/paper/PAPER_DRAFT_EN.md}
BIB_PATH=${2:-references.bib}

if [[ ! -f "$DRAFT_PATH" ]]; then
  echo "[ERROR] Draft not found: $DRAFT_PATH" >&2
  exit 2
fi
if [[ ! -f "$BIB_PATH" ]]; then
  echo "[ERROR] Bib not found: $BIB_PATH" >&2
  exit 2
fi

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

CITE_KEYS="$TMP_DIR/cite_keys.txt"
BIB_KEYS="$TMP_DIR/bib_keys.txt"
MISSING_KEYS="$TMP_DIR/missing_keys.txt"

# Extract citation keys from \cite{...}, \citet{...}, \citep{...}, etc.
# Split multi-key cites on commas.
perl -0777 -ne 'while(/\\cite\w*\{([^}]+)\}/g){print "$1\n"}' "$DRAFT_PATH" \
  | tr ',' '\n' \
  | sed 's/^ *//;s/ *$//' \
  | grep -v '^$' \
  | sort -u > "$CITE_KEYS"

# Extract entry keys from BibTeX.
perl -ne 'if(/^@\w+\{([^,]+),/){print "$1\n"}' "$BIB_PATH" \
  | sort -u > "$BIB_KEYS"

comm -23 "$CITE_KEYS" "$BIB_KEYS" > "$MISSING_KEYS" || true

if [[ -s "$MISSING_KEYS" ]]; then
  echo "[FAIL] Missing BibTeX entries for the following citation keys (in $DRAFT_PATH but not in $BIB_PATH):" >&2
  sed 's/^/  - /' "$MISSING_KEYS" >&2
  exit 1
fi

echo "[OK] All citation keys in $DRAFT_PATH exist in $BIB_PATH (n=$(wc -l < "$CITE_KEYS" | tr -d ' '))."