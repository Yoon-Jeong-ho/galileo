#!/usr/bin/env python3
"""Build tier1_*_survival_summary_*.csv from multiple per-seed paper_exports.

This matches the existing artifact schema used for cross-family plots:
model,persona,seeds,survival_r5_mean,survival_r5_std,
  delta_survival_r5_mean,delta_survival_r5_std,
  delta_fail_r1_mean,delta_fail_r1_std

Inputs are *paper_exports directories* (one per seed). Each paper_exports must
contain:
  - survival_curve.csv
  - turn_of_failure.csv

We treat Survival@5 as survival_rate at round==5 (converted to [0,1]).
We treat Fail@1 as the aggregate fail_at_1 rate over all test_name rows.
Deltas are computed per-seed relative to the persona 'neutral_reask_control'.

Stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_std(xs: Iterable[float]) -> Tuple[float, float]:
    xs = [float(x) for x in xs]
    if not xs:
        return (0.0, 0.0)
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return (m, 0.0)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return (m, math.sqrt(var))


@dataclass(frozen=True)
class SeedMetrics:
    survival_r5: Dict[str, float]  # persona -> [0,1]
    fail_r1: Dict[str, float]      # persona -> [0,1]


def load_seed_metrics(paper_exports: Path) -> SeedMetrics:
    surv_rows = read_csv(paper_exports / "survival_curve.csv")
    survival_r5: Dict[str, float] = {}
    for r in surv_rows:
        if int(r["round"]) != 5:
            continue
        p = r["persona"]
        survival_r5[p] = float(r["survival_rate"]) / 100.0

    tof_rows = read_csv(paper_exports / "turn_of_failure.csv")
    # Aggregate fail_at_1 over all test rows.
    num: Dict[str, int] = {}
    den: Dict[str, int] = {}
    for r in tof_rows:
        if int(r["fail_turn"]) != 1:
            continue
        p = r["persona"]
        num[p] = num.get(p, 0) + int(r["count"])
        den[p] = den.get(p, 0) + int(r["total"])

    fail_r1 = {p: (num[p] / den[p] if den[p] else 0.0) for p in num.keys()}

    # Ensure all personas present in both dicts (fill 0.0 if missing).
    personas = set(survival_r5) | set(fail_r1)
    for p in personas:
        survival_r5.setdefault(p, 0.0)
        fail_r1.setdefault(p, 0.0)

    return SeedMetrics(survival_r5=survival_r5, fail_r1=fail_r1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id string")
    ap.add_argument("--seeds", required=True, help="comma list, e.g., 1,2")
    ap.add_argument("--paper_exports", nargs="+", required=True, help="Paths to per-seed paper_exports/")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    if len(seeds) != len(args.paper_exports):
        raise SystemExit(f"seeds count ({len(seeds)}) != paper_exports count ({len(args.paper_exports)})")

    per_seed: List[SeedMetrics] = [load_seed_metrics(Path(p)) for p in args.paper_exports]

    # Union of personas across seeds.
    personas = sorted({p for sm in per_seed for p in sm.survival_r5.keys()})
    if "neutral_reask_control" not in personas:
        raise SystemExit("missing persona 'neutral_reask_control' in survival_curve.csv")

    rows_out: List[dict] = []

    for persona in personas:
        surv_vals = [sm.survival_r5.get(persona, 0.0) for sm in per_seed]
        fail_vals = [sm.fail_r1.get(persona, 0.0) for sm in per_seed]

        surv_mean, surv_std = mean_std(surv_vals)

        # Deltas per seed relative to NRC.
        delta_surv_vals = [
            sm.survival_r5.get(persona, 0.0) - sm.survival_r5.get("neutral_reask_control", 0.0)
            for sm in per_seed
        ]
        delta_fail_vals = [
            sm.fail_r1.get(persona, 0.0) - sm.fail_r1.get("neutral_reask_control", 0.0)
            for sm in per_seed
        ]
        dsm, dss = mean_std(delta_surv_vals)
        dfm, dfs = mean_std(delta_fail_vals)

        rows_out.append(
            {
                "model": args.model,
                "persona": persona,
                "seeds": ",".join(seeds),
                "survival_r5_mean": f"{surv_mean:.8f}",
                "survival_r5_std": f"{surv_std:.12f}",
                "delta_survival_r5_mean": f"{dsm:.8f}",
                "delta_survival_r5_std": f"{dss:.12f}",
                "delta_fail_r1_mean": f"{dfm:.8f}",
                "delta_fail_r1_std": f"{dfs:.12f}",
            }
        )

    # Write stable order: NRC first, then others alpha.
    rows_out.sort(key=lambda r: (0 if r["persona"] == "neutral_reask_control" else 1, r["persona"]))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "model",
        "persona",
        "seeds",
        "survival_r5_mean",
        "survival_r5_std",
        "delta_survival_r5_mean",
        "delta_survival_r5_std",
        "delta_fail_r1_mean",
        "delta_fail_r1_std",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
