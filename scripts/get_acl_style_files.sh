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

REF="${1:-main}"
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
TMP_DIR="$(mktemp -d -t acl-style-files.XXXXXX)"

cleanup() { rm -f "$TMP_ZIP"; rm -rf "$TMP_DIR"; }
trap cleanup EXIT

if ! command -v curl >/dev/null 2>&1; then
  echo "[ERR] curl not found" >&2
  exit 2
fi

echo "[INFO] downloading: $URL" >&2
curl -fsSL "$URL" -o "$TMP_ZIP"

TMP_ZIP="$TMP_ZIP" TMP_DIR="$TMP_DIR" OUT_DIR="$OUT_DIR" python3 - <<'PY'
import os
import zipfile
from pathlib import Path

zip_path = Path(os.environ["TMP_ZIP"])
tmp_dir = Path(os.environ["TMP_DIR"])
out_dir = Path(os.environ["OUT_DIR"])

with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(tmp_dir)

# GitHub archives typically contain a single top-level directory like acl-style-files-<ref>/...
children = [p for p in tmp_dir.iterdir() if p.name not in ("__MACOSX",)]
if len(children) == 1 and children[0].is_dir():
    root = children[0]
else:
    root = tmp_dir

# Copy (not move) to keep tmp_dir cleanup simple.
# We want OUT_DIR to directly contain the style files (no nested archive root).
for src in root.iterdir():
    dst = out_dir / src.name
    if src.is_dir():
        # Python 3.8+ shutil.copytree has dirs_exist_ok
        import shutil
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.write_bytes(src.read_bytes())

print(f"[OK] extracted to: {out_dir} (flattened)")
PY

echo "[NOTE] For reproducibility, prefer pinning REF to a commit SHA or tag rather than 'main'." >&2
