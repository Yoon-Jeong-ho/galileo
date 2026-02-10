#!/usr/bin/env bash
set -euo pipefail

# Download a user-local Inkscape AppImage (no sudo required) for SVG->PDF conversion.
# Target path (default): tools/inkscape/inkscape.AppImage
#
# This script pins a known-good stable release and verifies sha256.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT/tools/inkscape"
OUT="$OUT_DIR/inkscape.AppImage"

# Inkscape 1.4.3 AppImage (x86_64) from inkscape.org (stable release).
URL="https://inkscape.org/gallery/item/58919/Inkscape-0d15f75-x86_64.AppImage"
SHA256_EXPECTED="cb65ccb4bb070d9b8a61483e5cbfd0340c1e10c048db4a677053caad14be5e69"

mkdir -p "$OUT_DIR"

tmp="$OUT.part"

fetch() {
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 -o "$tmp" "$URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$tmp" "$URL"
  else
    echo "[ERR] Need curl or wget to download AppImage" >&2
    exit 2
  fi
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    python3 - <<PY
import hashlib
p=r"$1"
h=hashlib.sha256()
with open(p,'rb') as f:
    for b in iter(lambda: f.read(1024*1024), b''):
        h.update(b)
print(h.hexdigest())
PY
  fi
}

if [[ -x "$OUT" ]]; then
  echo "[OK] already present: $OUT"
  exit 0
fi

echo "[..] downloading Inkscape AppImage -> $OUT"
fetch

got="$(sha256_file "$tmp")"
if [[ "$got" != "$SHA256_EXPECTED" ]]; then
  echo "[ERR] sha256 mismatch" >&2
  echo "      expected: $SHA256_EXPECTED" >&2
  echo "      got:      $got" >&2
  rm -f "$tmp"
  exit 3
fi

mv "$tmp" "$OUT"
chmod +x "$OUT"

echo "[OK] installed AppImage: $OUT"

# Quick smoke test: print version (does not require GUI).
"$OUT" --version || true
