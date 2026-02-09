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

    # JSON parse checks + minimal schema validation
    meta_obj = None
    runner_obj = None

    jn = exports_dir / "metadata.json"
    if jn.exists():
        try:
            meta_obj = read_json(jn)
        except Exception as ex:
            errors.append(f"invalid json {jn}: {ex}")

    if runner_meta.exists():
        try:
            runner_obj = read_json(runner_meta)
        except Exception as ex:
            errors.append(f"invalid json {runner_meta}: {ex}")

    if runner_obj is not None:
        required_keys = [
            "generated_at",
            "gpu_list",
            "tensor_parallel_size",
            "num_samples",
            "max_model_len",
            "max_tokens",
            "conda_env",
            "model",
            "seed",
        ]
        for k in required_keys:
            if k not in runner_obj:
                errors.append(f"{runner_meta} missing required key: {k}")

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


def _parity_check_runner_metadata(exports_dirs: list[Path]) -> list[str]:
    """Check that repeated runs for the same (model, seed, tag/temp) share identical infra/decoding settings.

    This is meant to catch accidental mismatches between persona/control (or reruns) within the same results tree.
    """
    errors: list[str] = []

    groups: dict[tuple, list[tuple[Path, dict]]] = {}
    for d in exports_dirs:
        rm = d / "runner_metadata.json"
        if not rm.exists():
            continue
        try:
            obj = read_json(rm)
        except Exception:
            continue

        key = (
            obj.get("model"),
            obj.get("seed"),
            obj.get("tag"),
            obj.get("greedy_temperature"),
        )
        groups.setdefault(key, []).append((d, obj))

    # Compare within groups that have more than 1 member.
    cmp_keys = [
        "gpu_list",
        "tensor_parallel_size",
        "num_samples",
        "max_model_len",
        "max_tokens",
        "conda_env",
    ]

    for key, items in groups.items():
        if len(items) <= 1:
            continue
        base_dir, base_obj = items[0]
        for other_dir, other_obj in items[1:]:
            for k in cmp_keys:
                if base_obj.get(k) != other_obj.get(k):
                    errors.append(
                        f"runner_metadata parity mismatch for group={key}: key={k} {base_dir}={base_obj.get(k)} vs {other_dir}={other_obj.get(k)}"
                    )

    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument(
        "--require_control",
        action="store_true",
        help="Fail if survival_curve.csv/turn_of_failure.csv does not contain persona=neutral_reask_control",
    )
    ap.add_argument(
        "--check_runner_parity",
        action="store_true",
        help="Check that repeated runs for the same (model, seed, tag/temp) share identical runner settings",
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

    if args.check_runner_parity:
        errs = _parity_check_runner_metadata(exports_dirs)
        if errs:
            any_errors = True
            eprint("\n[FAIL] runner_metadata parity")
            for msg in errs:
                eprint(f"  - {msg}")
        else:
            print("[OK] runner_metadata parity")

    return 2 if any_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
