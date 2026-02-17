#!/usr/bin/env python3
"""Check that all \\cite{...} keys used in repo exist in references.bib.

Usage:
  python3 tools/check_bibkeys.py [--root <dir>] [--bib <path>] [--ext md tex]

Default root is docs/paper plus tmp/emnlp2023_smoketest (to catch LaTeX scaffolds).

This is a lightweight, local-only sanity check to avoid last-minute LaTeX build failures.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import Iterable


CITE_RE = re.compile(r"\\cite\{([^}]+)\}")
BIB_ENTRY_RE = re.compile(r"@[^{]+\{\s*([^,\s]+)\s*,")


def iter_files(root: pathlib.Path, exts: Iterable[str]) -> Iterable[pathlib.Path]:
    exts = {e.lower().lstrip('.') for e in exts}
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower().lstrip('.') in exts:
            yield p


def extract_cite_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for m in CITE_RE.finditer(text):
        for part in m.group(1).split(','):
            k = part.strip()
            if not k:
                continue
            # Allow placeholder cites like \cite{...} in scaffolds without failing the check.
            if re.fullmatch(r"\.+", k):
                continue
            keys.add(k)
    return keys


def extract_bib_keys(bib_text: str) -> set[str]:
    return set(BIB_ENTRY_RE.findall(bib_text))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', action='append', default=[], help='Root dir(s) to scan (repeatable)')
    ap.add_argument('--bib', default='references.bib', help='BibTeX file path')
    ap.add_argument('--ext', action='append', default=['md', 'tex'], help='File extension(s) to scan (repeatable)')
    args = ap.parse_args()

    roots = [pathlib.Path(r) for r in (args.root or ['docs/paper', 'tmp/emnlp2023_smoketest'])]
    bib_path = pathlib.Path(args.bib)

    if not bib_path.exists():
        print(f"ERROR: bib file not found: {bib_path}", file=sys.stderr)
        return 2

    bib_keys = extract_bib_keys(bib_path.read_text(encoding='utf-8', errors='replace'))
    used_keys: set[str] = set()

    for root in roots:
        if not root.exists():
            continue
        for f in iter_files(root, args.ext):
            try:
                txt = f.read_text(encoding='utf-8', errors='replace')
            except Exception as e:
                print(f"WARN: failed reading {f}: {e}", file=sys.stderr)
                continue
            used_keys |= extract_cite_keys(txt)

    missing = sorted(k for k in used_keys if k not in bib_keys)

    if missing:
        print('Missing bib entries for cite keys:')
        for k in missing:
            print(f'  - {k}')
        print(f"\nTotal missing: {len(missing)}")
        return 1

    print(f"OK: all {len(used_keys)} cite keys found in {bib_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
