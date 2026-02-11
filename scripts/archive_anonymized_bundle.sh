#!/usr/bin/env bash
set -euo pipefail

# Archive a staged anonymized bundle directory into a tar.gz (and optionally zip) + sha256.
#
# Usage:
#   ./scripts/archive_anonymized_bundle.sh <STAGED_DIR> [OUT_PREFIX]
#
# Options (env vars):
#   MAKE_ZIP=1   additionally create <OUT_PREFIX>.zip + <OUT_PREFIX>.zip.sha256 (default 0)
#
# Examples:
#   ./scripts/archive_anonymized_bundle.sh tmp/anonymized_bundle_post_archive_pdfcheck
#   MAKE_ZIP=1 ./scripts/archive_anonymized_bundle.sh tmp/anonymized_bundle_post_archive_pdfcheck tmp/galileo_anonymized_bundle_20260211

STAGED_DIR=${1:-}
OUT_PREFIX=${2:-}

if [[ -z "$STAGED_DIR" ]]; then
  echo "[ERR] need STAGED_DIR" >&2
  exit 2
fi
if [[ ! -d "$STAGED_DIR" ]]; then
  echo "[ERR] not a directory: $STAGED_DIR" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "$OUT_PREFIX" ]]; then
  base="$(basename "$STAGED_DIR")"
  OUT_PREFIX="$ROOT/tmp/${base}"
fi

MAKE_ZIP="${MAKE_ZIP:-0}"

TAR_PATH="${OUT_PREFIX}.tar.gz"
SHA_PATH="${OUT_PREFIX}.tar.gz.sha256"
ZIP_PATH="${OUT_PREFIX}.zip"
ZIP_SHA_PATH="${OUT_PREFIX}.zip.sha256"

# Create tar.gz (store relative paths)
parent="$(cd "$(dirname "$STAGED_DIR")" && pwd)"
name="$(basename "$STAGED_DIR")"

rm -f "$TAR_PATH" "$SHA_PATH"

# -C parent, archive the folder name
# --numeric-owner avoids leaking local usernames in tar metadata
# --mtime=0 gives reproducible-ish timestamps when GNU tar supports it (best-effort)
TAR_CMD=(tar -C "$parent" --numeric-owner -czf "$TAR_PATH" "$name")
"${TAR_CMD[@]}"

sha_file() {
  local in_path="$1"
  local out_path="$2"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$in_path" | awk '{print $1}' > "$out_path"
  else
    python3 - <<PY
import hashlib
p=r"$in_path"
h=hashlib.sha256()
with open(p,'rb') as f:
    for b in iter(lambda: f.read(1024*1024), b''):
        h.update(b)
print(h.hexdigest())
PY
  fi
}

sha_file "$TAR_PATH" "$SHA_PATH"

echo "[OK] wrote: $TAR_PATH"
echo "[OK] sha256: $(cat "$SHA_PATH")"

echo "[INFO] To verify:"
echo "  sha256sum -c <(echo \"$(cat "$SHA_PATH")  $TAR_PATH\")"

if [[ "$MAKE_ZIP" == "1" ]]; then
  rm -f "$ZIP_PATH" "$ZIP_SHA_PATH"
  if command -v zip >/dev/null 2>&1; then
    # Zip from parent so paths are relative.
    (cd "$parent" && zip -qr "$ROOT/${ZIP_PATH#$ROOT/}" "$name")
  else
    # Fallback: stdlib Python zipfile (slower but avoids extra deps)
    python3 - <<PY
import os
import zipfile
from pathlib import Path
parent=Path(r"$parent")
name=r"$name"
out=Path(r"$ZIP_PATH")
root = parent / name
with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for p in root.rglob('*'):
        arc = p.relative_to(parent).as_posix()
        if p.is_dir():
            # zipfile doesn't require explicit dir entries
            continue
        z.write(p, arc)
PY
  fi

  sha_file "$ZIP_PATH" "$ZIP_SHA_PATH"
  echo "[OK] wrote: $ZIP_PATH"
  echo "[OK] sha256: $(cat "$ZIP_SHA_PATH")"
  echo "[INFO] To verify:"
  echo "  sha256sum -c <(echo \"$(cat "$ZIP_SHA_PATH")  $ZIP_PATH\")"
fi
