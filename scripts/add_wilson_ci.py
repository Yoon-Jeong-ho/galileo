#!/usr/bin/env python3
"""Add Wilson score binomial confidence intervals to a CSV.

Motivation
----------
Many paper-facing exports (e.g., survival_curve.csv) report rates as k/n.
This helper augments such tables with a simple, assumption-light 95% CI
(Wilson score interval) without rerunning any experiments.

Usage examples
--------------
# Survival curves (k=survived, n=total)
python3 scripts/add_wilson_ci.py \
  --in_csv  path/to/paper_exports/survival_curve.csv \
  --out_csv path/to/paper_exports/survival_curve_wilson95.csv \
  --k_col survived --n_col total

# Turn-of-failure histograms (k=count, n=total)
python3 scripts/add_wilson_ci.py \
  --in_csv  path/to/paper_exports/turn_of_failure.csv \
  --out_csv path/to/paper_exports/turn_of_failure_wilson95.csv \
  --k_col count --n_col total

Notes
-----
- stdlib-only.
- The output keeps all original columns and appends:
    ci_low, ci_high, ci_level
  in the same units as the input rate (i.e., percent if you used percent).
"""

import argparse
import csv
import math
from pathlib import Path


def z_from_level(level: float) -> float:
    # Common levels only; keep stdlib-only.
    # 95% two-sided ≈ 1.959963984540054
    if abs(level - 0.95) < 1e-12:
        return 1.959963984540054
    if abs(level - 0.90) < 1e-12:
        return 1.6448536269514722
    if abs(level - 0.99) < 1e-12:
        return 2.5758293035489004
    raise ValueError(f"Unsupported level={level}. Use 0.90, 0.95, or 0.99.")


def wilson_interval(k: float, n: float, z: float):
    """Wilson score interval for a binomial proportion.

    Returns (low, high) in [0,1].
    """
    if n <= 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n))
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return (low, high)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--k_col", required=True, help="success count column (k)")
    ap.add_argument("--n_col", required=True, help="total count column (n)")
    ap.add_argument("--level", type=float, default=0.95, help="CI level (0.90/0.95/0.99)")
    ap.add_argument(
        "--rate_col",
        default=None,
        help=(
            "Optional: existing rate column to infer scaling (e.g., percent vs fraction). "
            "If provided, we preserve its scale when writing ci_low/high."
        ),
    )
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)
    z = z_from_level(args.level)

    with in_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not rows:
        raise SystemExit(f"No rows found in {in_path}")

    scale = 1.0
    if args.rate_col is not None:
        # Heuristic: if the provided rate col looks like percent (>1 for typical rates), treat as percent.
        # We inspect the first non-empty value.
        for r in rows:
            v = (r.get(args.rate_col) or "").strip()
            if not v:
                continue
            try:
                fv = float(v)
            except ValueError:
                continue
            if fv > 1.0:
                scale = 100.0
            break

    for r in rows:
        k = float(r[args.k_col])
        n = float(r[args.n_col])
        lo, hi = wilson_interval(k, n, z)
        r["ci_low"] = f"{lo * scale:.6f}"
        r["ci_high"] = f"{hi * scale:.6f}"
        r["ci_level"] = f"{args.level:.2f}"

    extra = ["ci_low", "ci_high", "ci_level"]
    for c in extra:
        if c not in fieldnames:
            fieldnames.append(c)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
