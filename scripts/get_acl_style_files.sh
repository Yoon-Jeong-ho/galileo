#!/usr/bin/env bash
set -euo pipefail

# Download ACL (incl. EMNLP) official LaTeX style files into the repo (for local LaTeX builds).
# Source of truth: https://github.com/acl-org/acl-style-files
#
# Usage:
#   bash scripts/get_acl_style_files.sh [REF]
#
# REF:
#   - git ref/sha/tag (default: main)
#
# Output:
#   docs/paper/acl_style_files/<REF>/... (unzipped)

REF="${1:-master}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT/docs/paper/acl_style_files/${REF}"

mkdir -p "$OUT_DIR"

# GitHub archive URLs require the refs/heads prefix for branch names.
if [[ "$REF" == "main" || "$REF" == "master" ]]; then
  URL="https://github.com/acl-org/acl-style-files/archive/refs/heads/${REF}.zip"
else
  # tags/SHAs typically work without refs prefix
  URL="https://github.com/acl-org/acl-style-files/archive/${REF}.zip"
fi
TMP_ZIP="$(mktemp -t acl-style-files.XXXXXX.zip)"

cleanup() { rm -f "$TMP_ZIP"; }
trap cleanup EXIT

if ! command -v curl >/dev/null 2>&1; then
  echo "[ERR] curl not found" >&2
  exit 2
fi

echo "[INFO] downloading: $URL" >&2
curl -fsSL "$URL" -o "$TMP_ZIP"

python3 - <<PY
import zipfile
from pathlib import Path
zip_path = Path("$TMP_ZIP")
out_dir = Path("$OUT_DIR")
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(out_dir)
print(f"[OK] extracted to: {out_dir}")
PY

echo "[NOTE] For reproducibility, prefer pinning REF to a commit SHA or tag rather than 'main'." >&2
