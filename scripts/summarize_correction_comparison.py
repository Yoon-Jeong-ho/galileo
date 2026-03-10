#!/usr/bin/env python3
"""Summarize evidence-bearing vs grounded-correction runs from analysis CSV.

Input: summary_by_arm.csv produced by scripts/analyze_baseline_suite.py
Output: one row per dataset comparing authority/control metrics across run labels.
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
    ap.add_argument("--evidence_label", required=True)
    ap.add_argument("--grounded_label", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    rows = read_csv(Path(args.summary_csv))
    keyed = {(r["run_label"], r["dataset"], r["persona"]): r for r in rows}
    datasets = sorted({r["dataset"] for r in rows})
    out_rows = []
    for dataset in datasets:
        ev_auth = keyed.get((args.evidence_label, dataset, "authority_claim"))
        gr_auth = keyed.get((args.grounded_label, dataset, "authority_claim"))
        ev_ctl = keyed.get((args.evidence_label, dataset, "control_reask"))
        gr_ctl = keyed.get((args.grounded_label, dataset, "control_reask"))
        if not (ev_auth and gr_auth and ev_ctl and gr_ctl):
            continue
        out_rows.append(
            {
                "dataset": dataset,
                "evidence_authority_survival_r5": ev_auth["survival_r5"],
                "grounded_authority_survival_r5": gr_auth["survival_r5"],
                "delta_authority_survival_r5": f"{float(gr_auth['survival_r5']) - float(ev_auth['survival_r5']):.6f}",
                "evidence_authority_fail1": ev_auth["fail1"],
                "grounded_authority_fail1": gr_auth["fail1"],
                "delta_authority_fail1": f"{float(gr_auth['fail1']) - float(ev_auth['fail1']):.6f}",
                "evidence_authority_recovery_rate": ev_auth["recovery_rate"],
                "grounded_authority_recovery_rate": gr_auth["recovery_rate"],
                "evidence_authority_post_recovery_acc": ev_auth["post_recovery_acc"],
                "grounded_authority_post_recovery_acc": gr_auth["post_recovery_acc"],
                "delta_authority_post_recovery_acc": f"{float(gr_auth['post_recovery_acc']) - float(ev_auth['post_recovery_acc']):.6f}",
                "evidence_control_survival_r5": ev_ctl["survival_r5"],
                "grounded_control_survival_r5": gr_ctl["survival_r5"],
            }
        )

    write_csv(
        Path(args.out_csv),
        [
            "dataset",
            "evidence_authority_survival_r5",
            "grounded_authority_survival_r5",
            "delta_authority_survival_r5",
            "evidence_authority_fail1",
            "grounded_authority_fail1",
            "delta_authority_fail1",
            "evidence_authority_recovery_rate",
            "grounded_authority_recovery_rate",
            "evidence_authority_post_recovery_acc",
            "grounded_authority_post_recovery_acc",
            "delta_authority_post_recovery_acc",
            "evidence_control_survival_r5",
            "grounded_control_survival_r5",
        ],
        out_rows,
    )
    print(f"Wrote: {args.out_csv} ({len(out_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
