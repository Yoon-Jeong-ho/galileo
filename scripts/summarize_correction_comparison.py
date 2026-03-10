#!/usr/bin/env python3
"""Summarize evidence-bearing vs grounded-correction runs from analysis CSV.

Input: summary_by_arm.csv produced by scripts/analyze_baseline_suite.py
Output: one row per dataset comparing authority/control metrics across two run labels.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--variant_a_label", required=True)
    ap.add_argument("--variant_b_label", required=True)
    ap.add_argument("--variant_a_name", required=True, help="column prefix label, e.g. evidence")
    ap.add_argument("--variant_b_name", required=True, help="column prefix label, e.g. grounded")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    rows = read_csv(Path(args.summary_csv))
    keyed = {(r["run_label"], r["dataset"], r["persona"]): r for r in rows}
    datasets = sorted({r["dataset"] for r in rows})
    out_rows = []
    for dataset in datasets:
        a_auth = keyed.get((args.variant_a_label, dataset, "authority_claim"))
        b_auth = keyed.get((args.variant_b_label, dataset, "authority_claim"))
        a_ctl = keyed.get((args.variant_a_label, dataset, "control_reask"))
        b_ctl = keyed.get((args.variant_b_label, dataset, "control_reask"))
        if not (a_auth and b_auth and a_ctl and b_ctl):
            continue
        row = {"dataset": dataset}
        a_name = args.variant_a_name
        b_name = args.variant_b_name
        row[f"{a_name}_authority_survival_r5"] = a_auth["survival_r5"]
        row[f"{b_name}_authority_survival_r5"] = b_auth["survival_r5"]
        row["delta_authority_survival_r5"] = f"{float(b_auth['survival_r5']) - float(a_auth['survival_r5']):.6f}"
        row[f"{a_name}_authority_fail1"] = a_auth["fail1"]
        row[f"{b_name}_authority_fail1"] = b_auth["fail1"]
        row["delta_authority_fail1"] = f"{float(b_auth['fail1']) - float(a_auth['fail1']):.6f}"
        row[f"{a_name}_authority_recovery_rate"] = a_auth["recovery_rate"]
        row[f"{b_name}_authority_recovery_rate"] = b_auth["recovery_rate"]
        row[f"{a_name}_authority_post_recovery_acc"] = a_auth["post_recovery_acc"]
        row[f"{b_name}_authority_post_recovery_acc"] = b_auth["post_recovery_acc"]
        row["delta_authority_post_recovery_acc"] = f"{float(b_auth['post_recovery_acc']) - float(a_auth['post_recovery_acc']):.6f}"
        row[f"{a_name}_control_survival_r5"] = a_ctl["survival_r5"]
        row[f"{b_name}_control_survival_r5"] = b_ctl["survival_r5"]
        out_rows.append(row)

    fieldnames = ["dataset"]
    for prefix in [args.variant_a_name, args.variant_b_name]:
        fieldnames.extend(
            [
                f"{prefix}_authority_survival_r5",
                f"{prefix}_authority_fail1",
                f"{prefix}_authority_recovery_rate",
                f"{prefix}_authority_post_recovery_acc",
                f"{prefix}_control_survival_r5",
            ]
        )
    # Reorder with deltas near compared metrics.
    fieldnames = [
        "dataset",
        f"{args.variant_a_name}_authority_survival_r5",
        f"{args.variant_b_name}_authority_survival_r5",
        "delta_authority_survival_r5",
        f"{args.variant_a_name}_authority_fail1",
        f"{args.variant_b_name}_authority_fail1",
        "delta_authority_fail1",
        f"{args.variant_a_name}_authority_recovery_rate",
        f"{args.variant_b_name}_authority_recovery_rate",
        f"{args.variant_a_name}_authority_post_recovery_acc",
        f"{args.variant_b_name}_authority_post_recovery_acc",
        "delta_authority_post_recovery_acc",
        f"{args.variant_a_name}_control_survival_r5",
        f"{args.variant_b_name}_control_survival_r5",
    ]

    write_csv(Path(args.out_csv), fieldnames, out_rows)
    print(f"Wrote: {args.out_csv} ({len(out_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
