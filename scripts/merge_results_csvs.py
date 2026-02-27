#!/usr/bin/env python3
"""Merge root-level GALILEO CSV summaries across multiple results directories.

Why:
- Some long scale-up runs can stall mid-sweep.
- We can run each dataset (or subset) in a separate process/results_dir, then
  merge the resulting root CSV summaries by summing numerators/denominators.

This script merges:
- initial_accuracy.csv
- adversarial_survival.csv
- recovery_accuracy.csv

It is stdlib-only.

Usage:
  python3 scripts/merge_results_csvs.py \
    --out_dir results/merged_seed1_scaleup_1000_1024 \
    --results_dirs results/run_gsm8k results/run_arc results/run_triviaqa

Notes:
- The merge is additive: counts are summed for identical keys.
- Rates are recomputed from merged counts.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path


def _read_csv(path: Path):
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def merge_initial(results_dirs: list[Path], out_dir: Path) -> bool:
    acc = defaultdict(lambda: {"correct": 0, "total": 0})
    found = False
    for rd in results_dirs:
        p = rd / "initial_accuracy.csv"
        if not p.exists():
            continue
        found = True
        for r in _read_csv(p):
            key = (r["model"], r["test_name"])
            acc[key]["correct"] += int(r["correct"])
            acc[key]["total"] += int(r["total"])

    if not found:
        return False

    out_path = out_dir / "initial_accuracy.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "test_name", "correct", "total", "accuracy"])
        for (model, test) in sorted(acc.keys()):
            correct = acc[(model, test)]["correct"]
            total = acc[(model, test)]["total"]
            rate = (correct / total * 100.0) if total else 0.0
            writer.writerow([model, test, correct, total, f"{rate:.2f}"])
    return True


def merge_adv(results_dirs: list[Path], out_dir: Path) -> bool:
    acc = defaultdict(lambda: {"survived": 0, "total": 0})
    found = False
    for rd in results_dirs:
        p = rd / "adversarial_survival.csv"
        if not p.exists():
            continue
        found = True
        for r in _read_csv(p):
            key = (r["model"], r["test_name"], r["persona"], int(r["round"]))
            acc[key]["survived"] += int(r["survived"])
            acc[key]["total"] += int(r["total"])

    if not found:
        return False

    out_path = out_dir / "adversarial_survival.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "test_name", "persona", "round", "survived", "total", "survival_rate"])
        for (model, test, persona, rnd) in sorted(acc.keys()):
            survived = acc[(model, test, persona, rnd)]["survived"]
            total = acc[(model, test, persona, rnd)]["total"]
            rate = (survived / total * 100.0) if total else 0.0
            writer.writerow([model, test, persona, rnd, survived, total, f"{rate:.2f}"])
    return True


def merge_recovery(results_dirs: list[Path], out_dir: Path) -> bool:
    acc = defaultdict(lambda: {"recovered": 0, "total": 0})
    found = False
    for rd in results_dirs:
        p = rd / "recovery_accuracy.csv"
        if not p.exists():
            continue
        found = True
        for r in _read_csv(p):
            key = (r["model"], r["test_name"], r["persona"])
            acc[key]["recovered"] += int(r["recovered"])
            acc[key]["total"] += int(r["total"])

    if not found:
        return False

    out_path = out_dir / "recovery_accuracy.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "test_name", "persona", "recovered", "total", "recovery_rate"])
        for (model, test, persona) in sorted(acc.keys()):
            recovered = acc[(model, test, persona)]["recovered"]
            total = acc[(model, test, persona)]["total"]
            rate = (recovered / total * 100.0) if total else 0.0
            writer.writerow([model, test, persona, recovered, total, f"{rate:.2f}"])
    return True


def _copy_jsonls(results_dirs: list[Path], out_dir: Path, model_dir: str) -> None:
    """Copy per-example JSONLs into a merged directory so paper_export can compute TOF/flip_samples.

    We copy any *.jsonl under <results_dir>/<model_dir>/ into <out_dir>/<model_dir>/.

    If a filename collision occurs, we keep the first copy and warn if sizes differ.
    """

    dst_model = out_dir / model_dir
    dst_model.mkdir(parents=True, exist_ok=True)

    copied = 0
    for rd in results_dirs:
        src_model = rd / model_dir
        if not src_model.exists():
            raise SystemExit(f"--copy_jsonls requested but missing model_dir in results_dir: {src_model}")

        for src in sorted(src_model.glob("*.jsonl")):
            dst = dst_model / src.name
            if dst.exists():
                if dst.stat().st_size != src.stat().st_size:
                    print(f"[WARN] JSONL name collision with different size; keeping first: {dst}")
                continue
            shutil.copy2(src, dst)
            copied += 1

    print(f"[OK] copied {copied} JSONLs into: {dst_model}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--results_dirs", type=Path, nargs="+", required=True)
    ap.add_argument(
        "--copy_jsonls",
        action="store_true",
        help="Also copy per-example JSONLs into out_dir/<model_dir>/ so paper_export can produce TOF/flip_samples.",
    )
    ap.add_argument(
        "--model_dir",
        default=None,
        help="Model directory name under each results_dir (required if --copy_jsonls). Example: 'Mistral-7B-Instruct-v0.3'.",
    )
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    results_dirs = [p for p in args.results_dirs]

    ok_any = False
    ok_any |= merge_initial(results_dirs, out_dir)
    ok_any |= merge_adv(results_dirs, out_dir)
    ok_any |= merge_recovery(results_dirs, out_dir)

    if not ok_any:
        raise SystemExit("No input CSVs found under provided results_dirs")

    if args.copy_jsonls:
        if not args.model_dir:
            raise SystemExit("--model_dir is required when using --copy_jsonls")
        _copy_jsonls(results_dirs, out_dir, args.model_dir)

    print(f"[OK] wrote merged CSVs under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
