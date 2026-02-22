#!/usr/bin/env python3
"""Summarize Recovery@flip denominators (control_total/persona_total) for Table 1 rows.

Input:
- docs/paper/artifacts/table1_recovery_from_results_paper_YYYYMMDD.csv
  (per-alias recovery plus control_total/persona_total)

Output:
- docs/paper/artifacts/table1_recovery_denominators_YYYYMMDD.csv
  (per Table-1 row: mean±std of control_total and persona_total across the aliases/seeds)

This is intended as an Appendix helper so readers can gauge variance / small-n
for Recovery@flip.

Stdlib-only.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Agg:
    vals: List[float]

    def mean_std(self) -> Tuple[float, float]:
        if not self.vals:
            return (math.nan, math.nan)
        m = sum(self.vals) / len(self.vals)
        if len(self.vals) == 1:
            return (m, 0.0)
        v = sum((x - m) ** 2 for x in self.vals) / (len(self.vals) - 1)
        return (m, math.sqrt(v))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    # Must match Table~\ref{tab:main} row-to-alias mapping.
    row_to_aliases: Dict[str, List[str]] = {
        "Qwen2.5-7B (1--4)": [
            "qwen_control_seed1",
            "qwen_control_seed2",
            "qwen_control_seed3",
            "qwen_control_seed4",
        ],
        "Llama-3.1-8B (1--2)": [
            "llama_seed1",
            "llama_seed2",
        ],
        "Mistral-7B (1--2)": [
            "mistral_seed1",
            "mistral_seed2",
        ],
        "Llama-3.2-3B (1--2)": [
            "tier1_llama3_3b_seed1_20260212_030426",
            "tier1_llama3_3b_seed2_20260212_042339",
        ],
        "Phi-3-mini-4k (1--2)": [
            "tier1_phi3mini_seed1_20260217_011737",
            "tier1_phi3mini_seed2_20260217_033953",
        ],
        "Mistral-Nemo (1--2)": [
            "tier1_mistralnemo_seed1_20260217_173907",
            "tier1_mistralnemo_seed2_20260217_180951",
        ],
        "Qwen2.5-14B (1--2)": [
            "tier1_qwen2p5_14b_seed1_20260219_032551",
            "tier1_qwen2p5_14b_seed2_20260219_053824",
        ],
    }

    by_alias: Dict[str, dict] = {}
    with open(args.in_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            by_alias[row["alias"].strip()] = row

    out_rows = []
    for table_row, aliases in row_to_aliases.items():
        ctot = Agg([])
        ptot = Agg([])
        present = 0
        for a in aliases:
            if a not in by_alias:
                continue
            present += 1
            ctot.vals.append(float(by_alias[a]["control_total"]))
            ptot.vals.append(float(by_alias[a]["persona_total"]))

        cm, cs = ctot.mean_std()
        pm, ps = ptot.mean_std()
        out_rows.append(
            {
                "row": table_row,
                "num_aliases_present": str(present),
                "control_total_mean": f"{cm:.1f}" if not math.isnan(cm) else "",
                "control_total_std": f"{cs:.1f}" if not math.isnan(cs) else "",
                "persona_total_mean": f"{pm:.1f}" if not math.isnan(pm) else "",
                "persona_total_std": f"{ps:.1f}" if not math.isnan(ps) else "",
            }
        )

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "row",
                "num_aliases_present",
                "control_total_mean",
                "control_total_std",
                "persona_total_mean",
                "persona_total_std",
            ],
        )
        w.writeheader()
        w.writerows(out_rows)

    print(f"wrote {len(out_rows)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
