#!/usr/bin/env python3
"""Compute Recovery@flip (Control vs Persona) for Table 1 from results_paper.

Why this exists:
- Table~\ref{tab:main} needs Recovery@flip per model family.
- Our current paper_exports bundles in results_paper/ do not include recovery files.
- However, metadata.json records the original results_root, which typically contains
  recovery_accuracy.csv produced by run_experiment.py.

This script:
- Iterates over results_paper/<alias>/paper_exports/metadata.json
- Uses metadata["results_root"] to locate recovery_accuracy.csv
- Aggregates recovered/total across tests, then across personas:
  - C := NRC control persona (display name "Control Re-asking" in recovery_accuracy.csv)
  - P := all other personas (pressure) pooled (persona-weighted by total)
- Emits a per-alias CSV with:
    alias, nrc_recovery, persona_recovery, delta_recovery
  where values are in [0,1].

Note:
- This assumes recovery_accuracy.csv's `total` counts correspond to "flip cases".
  (i.e., recovery conditional on flipping). That matches our paper framing.

Usage (on nlp8):
  python3 scripts/make_table1_recovery_from_results_paper.py \
    --results_paper /data_x/aa007878/galileo/results_paper \
    --out_csv docs/paper/artifacts/table1_recovery_from_results_paper_YYYYMMDD.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


# Note: run_experiment.py writes persona names via get_persona_name(...), so the NRC
# persona appears as the *display name* "Control Re-asking" in recovery_accuracy.csv.
# (Survival exports use the normalized id neutral_reask_control, but recovery CSV is legacy.)
CONTROL_PERSONA = "Control Re-asking"


@dataclass
class Totals:
    recovered: int = 0
    total: int = 0

    def add(self, recovered: int, total: int) -> None:
        self.recovered += int(recovered)
        self.total += int(total)

    def rate(self) -> float:
        if self.total <= 0:
            return float("nan")
        return self.recovered / self.total


def read_recovery_csv(path: str) -> Iterable[dict]:
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            yield row


def compute_control_vs_persona(recovery_csv: str) -> Tuple[Totals, Totals]:
    c = Totals()
    p = Totals()
    for row in read_recovery_csv(recovery_csv):
        persona = row.get("persona", "").strip()
        recovered = int(float(row.get("recovered", "0")))
        total = int(float(row.get("total", "0")))
        if persona == CONTROL_PERSONA:
            c.add(recovered, total)
        else:
            p.add(recovered, total)
    return c, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_paper", required=True, help="Path to results_paper directory")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    rows = []

    for alias in sorted(os.listdir(args.results_paper)):
        alias_dir = os.path.join(args.results_paper, alias)
        meta_path = os.path.join(alias_dir, "paper_exports", "metadata.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, "r") as f:
            meta = json.load(f)
        results_root = meta.get("results_root")
        if not results_root:
            continue

        recovery_csv = os.path.join(results_root, "recovery_accuracy.csv")
        if not os.path.exists(recovery_csv):
            # Some legacy runs might store it under paper_exports/ (not standard).
            alt = os.path.join(alias_dir, "paper_exports", "recovery_accuracy.csv")
            if os.path.exists(alt):
                recovery_csv = alt
            else:
                continue

        c, p = compute_control_vs_persona(recovery_csv)
        rows.append(
            {
                "alias": alias,
                "nrc_recovery": c.rate(),
                "persona_recovery": p.rate(),
                "delta_recovery": p.rate() - c.rate(),
                "control_total": c.total,
                "persona_total": p.total,
            }
        )

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "alias",
                "nrc_recovery",
                "persona_recovery",
                "delta_recovery",
                "control_total",
                "persona_total",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"wrote {len(rows)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
