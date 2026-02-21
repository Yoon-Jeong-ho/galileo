#!/usr/bin/env python3
"""Generate a LaTeX-ready partial for Table 1 (Survival@5 + Fail@1).

Inputs:
  - docs/paper/artifacts/table1_from_results_paper_exports_YYYYMMDD.csv

This is intentionally *partial*: Recovery@flip and n0/seed are not included in the
current artifact, so we leave those columns as placeholders.

Usage:
  python3 scripts/make_table1_partial_from_results_paper_exports.py \
    --in_csv docs/paper/artifacts/table1_from_results_paper_exports_20260222.csv \
    --n0_csv docs/paper/artifacts/table1_n0_from_results_paper_exports_20260222.csv

Output:
  Prints LaTeX tabular rows with mean±std across seeds for each model family row.
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


def fmt(m: float, s: float, digits: int = 3) -> str:
    if math.isnan(m):
        return "--"
    if s == 0.0:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f}±{s:.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument(
        "--n0_csv",
        default=None,
        help="Optional CSV (alias,n0) derived from results_paper/*/paper_exports/survival_curve.csv",
    )
    ap.add_argument(
        "--recovery_csv",
        default=None,
        help=(
            "Optional CSV (alias,nrc_recovery,persona_recovery,delta_recovery,...) produced by "
            "scripts/make_table1_recovery_from_results_paper.py. Values are in [0,1]."
        ),
    )
    args = ap.parse_args()

    # Map Table-1 rows to aliases in the artifact CSV.
    # IMPORTANT: keep this in sync with docs/paper/PAPER_DRAFT_EN.md Table~\ref{tab:main}.
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

    by_alias: Dict[str, Dict[str, float]] = {}
    with open(args.in_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            alias = row["alias"].strip()
            by_alias[alias] = {
                k: float(v)
                for k, v in row.items()
                if k not in {"alias", "model", "status"} and v != ""
            }

    n0_by_alias: Dict[str, float] = {}
    if args.n0_csv:
        with open(args.n0_csv, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                n0_by_alias[row["alias"].strip()] = float(row["n0"])

    rec_by_alias: Dict[str, Dict[str, float]] = {}
    if args.recovery_csv:
        with open(args.recovery_csv, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                a = row["alias"].strip()
                rec_by_alias[a] = {
                    "c": float(row["nrc_recovery"]),
                    "p": float(row["persona_recovery"]),
                    "d": float(row["delta_recovery"]),
                }

    def fmt_pct(m: float, s: float) -> str:
        if math.isnan(m):
            return "--"
        return fmt(m * 100.0, s * 100.0, digits=1)

    # For each table row, aggregate per metric.
    for table_row, aliases in row_to_aliases.items():
        surv_c = Agg([])
        surv_p = Agg([])
        surv_d = Agg([])
        fail_c = Agg([])
        fail_p = Agg([])
        fail_d = Agg([])
        rec_c = Agg([])
        rec_p = Agg([])
        rec_d = Agg([])
        n0 = Agg([])

        missing = []
        for a in aliases:
            if a not in by_alias:
                missing.append(a)
                continue
            d = by_alias[a]
            surv_c.vals.append(d["nrc_survival_r5"])
            surv_p.vals.append(d["persona_survival_r5"])
            surv_d.vals.append(d["delta_survival_r5"])
            fail_c.vals.append(d["nrc_fail1"])
            fail_p.vals.append(d["persona_fail1"])
            fail_d.vals.append(d["delta_fail1"])
            if a in n0_by_alias:
                n0.vals.append(n0_by_alias[a])
            if a in rec_by_alias:
                rec_c.vals.append(rec_by_alias[a]["c"])
                rec_p.vals.append(rec_by_alias[a]["p"])
                rec_d.vals.append(rec_by_alias[a]["d"])

        sc_m, sc_s = surv_c.mean_std()
        sp_m, sp_s = surv_p.mean_std()
        sd_m, sd_s = surv_d.mean_std()
        fc_m, fc_s = fail_c.mean_std()
        fp_m, fp_s = fail_p.mean_std()
        fd_m, fd_s = fail_d.mean_std()
        rc_m, rc_s = rec_c.mean_std()
        rp_m, rp_s = rec_p.mean_std()
        rd_m, rd_s = rec_d.mean_std()
        n0_m, n0_s = n0.mean_std()

        miss_note = f" % MISSING: {', '.join(missing)}" if missing else ""
        print(
            f"{table_row} & "
            f"{fmt(sc_m, sc_s)} & {fmt(sp_m, sp_s)} & {fmt(sd_m, sd_s)} & "
            f"{fmt(fc_m, fc_s)} & {fmt(fp_m, fp_s)} & {fmt(fd_m, fd_s)} & "
            f"{fmt_pct(rc_m, rc_s)} & {fmt_pct(rp_m, rp_s)} & {fmt_pct(rd_m, rd_s)} & {fmt(n0_m, n0_s, digits=1)} \\\\" + miss_note
        )


if __name__ == "__main__":
    main()
