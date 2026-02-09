#!/usr/bin/env python3
"""Validate GALILEO paper export bundles (stdlib only).

This script is meant to catch silent omissions before plotting/writing:
- missing paper_exports/*.csv
- missing metadata.json / runner_metadata.json
- missing Neutral Re-asking Control rows in survival/TOF exports

Usage:
  python scripts/validate_paper_exports.py --results_root /path/to/results

Exit code:
  0 if all checks pass
  2 if any check fails
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REQUIRED_EXPORTS = [
    "survival_curve.csv",
    "turn_of_failure.csv",
    "flip_samples.csv",
    "metadata.json",
]


def eprint(*a):
    print(*a, file=sys.stderr)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_paper_exports_dirs(results_root: Path):
    # Convention: results/**/paper_exports/
    return sorted([p for p in results_root.rglob("paper_exports") if p.is_dir()])


def validate_one(exports_dir: Path, require_control: bool) -> list[str]:
    errors: list[str] = []

    # Basic presence checks
    for fn in REQUIRED_EXPORTS:
        p = exports_dir / fn
        if not p.exists():
            errors.append(f"missing {p}")

    # runner metadata is recommended; we treat it as required when present upstream runners.
    runner_meta = exports_dir / "runner_metadata.json"
    if not runner_meta.exists():
        errors.append(f"missing {runner_meta} (runner-side metadata)")

    # JSON parse checks
    for jn in [exports_dir / "metadata.json", runner_meta]:
        if jn.exists():
            try:
                read_json(jn)
            except Exception as ex:
                errors.append(f"invalid json {jn}: {ex}")

    # Control presence in exports
    if require_control:
        # survival curve
        surv = exports_dir / "survival_curve.csv"
        if surv.exists():
            try:
                rows = read_csv(surv)
                personas = {r.get("persona") for r in rows}
                if "neutral_reask_control" not in personas:
                    errors.append(
                        f"{surv} missing persona=neutral_reask_control (Neutral Re-asking Control)"
                    )
            except Exception as ex:
                errors.append(f"failed reading {surv}: {ex}")

        # turn-of-failure
        tof = exports_dir / "turn_of_failure.csv"
        if tof.exists():
            try:
                rows = read_csv(tof)
                personas = {r.get("persona") for r in rows}
                if "neutral_reask_control" not in personas:
                    errors.append(
                        f"{tof} missing persona=neutral_reask_control (Neutral Re-asking Control)"
                    )
            except Exception as ex:
                errors.append(f"failed reading {tof}: {ex}")

    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument(
        "--require_control",
        action="store_true",
        help="Fail if survival_curve.csv does not contain persona=neutral_reask_control",
    )
    args = ap.parse_args()

    root = Path(args.results_root)
    if not root.exists():
        eprint(f"ERROR: results_root not found: {root}")
        return 2

    exports_dirs = find_paper_exports_dirs(root)
    if not exports_dirs:
        eprint(f"ERROR: no paper_exports/ directories found under {root}")
        return 2

    any_errors = False
    for d in exports_dirs:
        errs = validate_one(d, require_control=bool(args.require_control))
        if errs:
            any_errors = True
            eprint(f"\n[FAIL] {d}")
            for msg in errs:
                eprint(f"  - {msg}")
        else:
            print(f"[OK] {d}")

    return 2 if any_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
