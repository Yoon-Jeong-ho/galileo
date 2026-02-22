#!/usr/bin/env python3
"""Print a LaTeX table body for Appendix denominators (Recovery@flip effective n).

Reads:
  docs/paper/artifacts/table1_recovery_denominators_YYYYMMDD.csv

Outputs lines:
  row & control_total_mean±std & persona_total_mean±std \\

Stdlib-only.
"""

from __future__ import annotations

import argparse
import csv


def fmt(ms: str, ss: str) -> str:
    if not ms:
        return "--"
    if not ss or float(ss) == 0.0:
        return f"{float(ms):.1f}"
    return f"{float(ms):.1f}±{float(ss):.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    args = ap.parse_args()

    with open(args.in_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            name = row["row"]
            c = fmt(row.get("control_total_mean", ""), row.get("control_total_std", ""))
            p = fmt(row.get("persona_total_mean", ""), row.get("persona_total_std", ""))
            print(f"{name} & {c} & {p} \\\\")


if __name__ == "__main__":
    main()
