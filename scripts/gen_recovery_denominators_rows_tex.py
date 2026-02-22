#!/usr/bin/env python3
"""Generate a LaTeX rows file for the Recovery@flip denominator appendix table.

Reads:
  docs/paper/artifacts/table1_recovery_denominators_YYYYMMDD.csv

Writes:
  docs/paper/latex_paper_emnlp2023/generated/recovery_denominators_rows.tex

Stdlib-only.
"""

from __future__ import annotations

import argparse
import csv
import os


def fmt(ms: str, ss: str) -> str:
    if not ms:
        return "--"
    m = float(ms)
    if not ss:
        return f"{m:.1f}"
    s = float(ss)
    if s == 0.0:
        return f"{m:.1f}"
    return f"{m:.1f}±{s:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument(
        "--out_tex",
        default="docs/paper/latex_paper_emnlp2023/generated/recovery_denominators_rows.tex",
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_tex), exist_ok=True)

    lines = []
    with open(args.in_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            name = row["row"]
            c = fmt(row.get("control_total_mean", ""), row.get("control_total_std", ""))
            p = fmt(row.get("persona_total_mean", ""), row.get("persona_total_std", ""))
            lines.append(f"{name} & {c} & {p} \\\\")

    with open(args.out_tex, "w") as f:
        f.write("% Auto-generated. Do not edit by hand.\n")
        f.write("% Source: " + args.in_csv + "\n")
        f.write("\n".join(lines) + "\n")

    print(f"wrote {len(lines)} rows -> {args.out_tex}")


if __name__ == "__main__":
    main()
