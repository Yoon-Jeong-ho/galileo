#!/usr/bin/env bash
set -euo pipefail

# Check that all \cite{...} keys referenced in paper markdown drafts exist in references.bib.
# Usage:
#   bash scripts/check_citations_vs_bib.sh [path/to/draft.md ...]
# Default drafts:
#   docs/paper/PAPER_DRAFT_EN.md docs/paper/PAPER_DRAFT_KO.md

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BIB_FILE="references.bib"
if [[ ! -f "$BIB_FILE" ]]; then
  echo "[ERROR] Missing $BIB_FILE (expected at repo root)." >&2
  exit 2
fi

if [[ "$#" -eq 0 ]]; then
  set -- docs/paper/PAPER_DRAFT_EN.md docs/paper/PAPER_DRAFT_KO.md
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

CITES_FILE="$TMP_DIR/cites.txt"
BIBKEYS_FILE="$TMP_DIR/bibkeys.txt"

# Extract bib keys
perl -ne 'if(/^\s*\@\w+\{([^,]+),/){print "$1\n"}' "$BIB_FILE" | sort -u > "$BIBKEYS_FILE"

# Extract cite keys from drafts
: > "$CITES_FILE"
for f in "$@"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Missing draft: $f" >&2
    exit 2
  fi
  perl -0777 -ne '
    while(/\\cite\{([^}]+)\}/g){
      $x=$1;
      for(split(/,\s*/,$x)){
        next if $_ eq "";
        print "$_\n";
      }
    }
  ' "$f" >> "$CITES_FILE"
done
sort -u "$CITES_FILE" -o "$CITES_FILE"

MISSING_FILE="$TMP_DIR/missing.txt"
comm -23 "$CITES_FILE" "$BIBKEYS_FILE" > "$MISSING_FILE" || true

if [[ -s "$MISSING_FILE" ]]; then
  echo "[FAIL] Missing BibTeX entries for the following citation keys:" >&2
  cat "$MISSING_FILE" >&2
  exit 1
fi

echo "[OK] All citation keys found in $BIB_FILE."
