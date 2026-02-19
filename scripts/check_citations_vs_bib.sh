#!/usr/bin/env bash
set -euo pipefail

# Check that all citation keys referenced in paper drafts exist in references.bib.
#
# We scan common LaTeX natbib-style commands (e.g., \cite{...}, \citet{...}, \citep{...})
# across Markdown drafts and any LaTeX skeleton files.
#
# Usage:
#   bash scripts/check_citations_vs_bib.sh [path/to/file.{md,tex} ...]
#
# Default inputs (if no args):
#   - docs/paper/PAPER_DRAFT_EN.md
#   - docs/paper/PAPER_DRAFT_KO.md
#   - all *.tex under docs/paper/

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BIB_FILE="references.bib"
if [[ ! -f "$BIB_FILE" ]]; then
  echo "[ERROR] Missing $BIB_FILE (expected at repo root)." >&2
  exit 2
fi

if [[ "$#" -eq 0 ]]; then
  # Default: scan everything paper-facing under docs/paper/.
  # (The repo does not always keep a single PAPER_DRAFT_*.md file around.)
  # Exclude the upstream EMNLP template, which may cite template-only entries.
  mapfile -t PAPER_FILES < <(
    find docs/paper -type f \( -name "*.tex" -o -name "*.md" \) \
      ! -path "docs/paper/emnlp_template/*" \
      | sort
  )
  set -- "${PAPER_FILES[@]}"
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
    # Match common natbib-style cite commands, with optional pre/post notes:
    #   \\cite{...}, \\citep{...}, \\citet{...}, \\citep[see][]{...}, etc.
    while(/\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}/g){
      $x=$1;
      for(split(/,\s*/,$x)){
        s/^\s+|\s+$//g;
        next if $_ eq "";
        # Common placeholder in drafts like \cite{...}
        next if $_ eq "...";
        # Be conservative: only keep plausible BibTeX keys.
        next unless /^[A-Za-z0-9:._-]+$/;
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
