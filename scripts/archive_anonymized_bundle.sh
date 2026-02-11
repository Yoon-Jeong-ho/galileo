#!/usr/bin/env bash
set -euo pipefail

# Archive a staged anonymized bundle directory into a tar.gz + sha256.
#
# Usage:
#   ./scripts/archive_anonymized_bundle.sh <STAGED_DIR> [OUT_PREFIX]
#
# Examples:
#   ./scripts/archive_anonymized_bundle.sh tmp/anonymized_bundle_post_archive_pdfcheck
#   ./scripts/archive_anonymized_bundle.sh tmp/anonymized_bundle_post_archive_pdfcheck tmp/galileo_anonymized_bundle_20260211

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

TAR_PATH="${OUT_PREFIX}.tar.gz"
SHA_PATH="${OUT_PREFIX}.tar.gz.sha256"

# Create tar.gz (store relative paths)
parent="$(cd "$(dirname "$STAGED_DIR")" && pwd)"
name="$(basename "$STAGED_DIR")"

rm -f "$TAR_PATH" "$SHA_PATH"

# -C parent, archive the folder name
# --numeric-owner avoids leaking local usernames in tar metadata
# --mtime=0 gives reproducible-ish timestamps when GNU tar supports it (best-effort)
TAR_CMD=(tar -C "$parent" --numeric-owner -czf "$TAR_PATH" "$name")
"${TAR_CMD[@]}"

# sha256
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$TAR_PATH" | awk '{print $1}' > "$SHA_PATH"
else
  python3 - <<PY
import hashlib
p=r"$TAR_PATH"
h=hashlib.sha256()
with open(p,'rb') as f:
    for b in iter(lambda: f.read(1024*1024), b''):
        h.update(b)
print(h.hexdigest())
PY
fi

echo "[OK] wrote: $TAR_PATH"
echo "[OK] sha256: $(cat "$SHA_PATH")"

echo "[INFO] To verify:"
echo "  sha256sum -c <(echo \"$(cat "$SHA_PATH")  $TAR_PATH\")"
