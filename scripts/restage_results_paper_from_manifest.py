#!/usr/bin/env python3
"""Restage results_paper/ from a simple manifest.

Why:
- results_paper/ is intentionally a paper-only *symlink root* and is often left untracked.
- On remote (nlp8), a `git clean -fd` will remove it, so we want a one-command rebuild.

Manifest format (TSV):
  alias\trun_dir

Where:
- alias = desired directory name under results_paper/
- run_dir = absolute path to the run directory that contains paper_exports/

Effect:
- Creates: results_paper/<alias>/paper_exports -> <run_dir>/paper_exports
- Optionally validates the root and writes results_paper/GLOBAL_VALIDATE.log

Usage:
  python3 scripts/restage_results_paper_from_manifest.py \
    --manifest docs/paper/results_paper_manifest.csv \
    --repo_root /data_x/aa007878/galileo \
    --validate

Notes:
- This script is stdlib-only and safe to run multiple times.
- It does NOT delete existing entries unless --clean is passed.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class Item:
    alias: str
    run_dir: str


def _mkdirp(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _safe_unlink(path: str) -> None:
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def read_manifest(path: str) -> list[Item]:
    items: list[Item] = []
    with open(path, "r", newline="") as f:
        # We use TSV because the repo .gitignore ignores *.csv under docs/.
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            alias = (row.get("alias") or "").strip()
            run_dir = (row.get("run_dir") or "").strip()
            if not alias or not run_dir:
                continue
            # Allow comment lines when users copy/paste the template.
            if alias.startswith("#"):
                continue
            items.append(Item(alias=alias, run_dir=run_dir))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--clean", action="store_true", help="Delete results_paper/ before staging")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    results_paper = os.path.join(repo_root, "results_paper")

    if args.clean and os.path.exists(results_paper):
        _safe_unlink(results_paper)

    _mkdirp(results_paper)

    items = read_manifest(args.manifest)
    if not items:
        raise SystemExit(f"No items found in manifest: {args.manifest}")

    for it in items:
        alias_dir = os.path.join(results_paper, it.alias)
        paper_exports = os.path.join(alias_dir, "paper_exports")
        target = os.path.join(it.run_dir, "paper_exports")

        if not os.path.isdir(target):
            raise SystemExit(f"Missing paper_exports for alias={it.alias}: {target}")

        _mkdirp(alias_dir)
        if os.path.lexists(paper_exports):
            _safe_unlink(paper_exports)
        os.symlink(target, paper_exports)

    if args.validate:
        cmd = [
            "python3",
            os.path.join(repo_root, "scripts", "validate_paper_exports.py"),
            "--results_root",
            results_paper,
            "--check_runner_parity",
        ]
        log_path = os.path.join(results_paper, "GLOBAL_VALIDATE.log")
        with open(log_path, "w") as f:
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            raise SystemExit(f"Validation failed (see {log_path})")

    print(f"Staged {len(items)} aliases under {results_paper}")


if __name__ == "__main__":
    main()
