#!/usr/bin/env python3
"""Make Table W: persona vs control summary.

This script is intentionally lightweight and artifact-driven.

Inputs are `paper_exports/` folders produced by `scripts/paper_export.py`:
- survival_curve.csv: persona, round, survived, total, survival_rate
- turn_of_failure.csv: persona, test_name, fail_turn, fail_turn_label, count, total, rate

We compute a compact comparison of CONTROL vs PERSONA pressure:
- Survival@R (default R=5) from survival_curve.csv
- Fail@1 and Never-fail from turn_of_failure.csv, aggregated across tasks

For PERSONA we provide two aggregates:
- weighted: weight each persona by its `total` in TOF table
- unweighted: simple mean across persona keys

Usage:
  python scripts/make_table_w_control_vs_persona.py \
    --control_exports /path/to/control/paper_exports \
    --persona_exports /path/to/persona/paper_exports \
    --out_csv /path/to/table_w.csv

Notes:
- This is seed-local. For multi-seed, run per seed then aggregate with scripts/aggregate_multiseed.py.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class SurvivalRow:
    persona: str
    round: int
    survived: int
    total: int
    survival_rate: float


@dataclass(frozen=True)
class TOFRow:
    persona: str
    test_name: str
    fail_turn: str
    fail_turn_label: str
    count: int
    total: int
    rate: float


def _read_survival(path: Path) -> List[SurvivalRow]:
    rows: List[SurvivalRow] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for d in r:
            rows.append(
                SurvivalRow(
                    persona=d["persona"],
                    round=int(d["round"]),
                    survived=int(d["survived"]),
                    total=int(d["total"]),
                    survival_rate=float(d["survival_rate"]),
                )
            )
    return rows


def _read_tof(path: Path) -> List[TOFRow]:
    rows: List[TOFRow] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for d in r:
            rows.append(
                TOFRow(
                    persona=d["persona"],
                    test_name=d["test_name"],
                    fail_turn=d["fail_turn"],
                    fail_turn_label=d["fail_turn_label"],
                    count=int(d["count"]),
                    total=int(d["total"]),
                    rate=float(d["rate"]),
                )
            )
    return rows


def _survival_at_r(rows: Iterable[SurvivalRow], r: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in rows:
        if row.round == r:
            out[row.persona] = row.survival_rate
    return out


def _aggregate_tof(rows: Iterable[TOFRow]) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """Return (persona, label) -> (count_sum, total_sum) aggregated across tasks."""
    agg: Dict[Tuple[str, str], Tuple[int, int]] = {}
    for row in rows:
        key = (row.persona, row.fail_turn_label)
        c, t = agg.get(key, (0, 0))
        agg[key] = (c + row.count, t + row.total)
    return agg


def _persona_keys(rows: Iterable[TOFRow]) -> List[str]:
    return sorted({r.persona for r in rows})


def _rate(count: int, total: int) -> float:
    return 0.0 if total <= 0 else 100.0 * count / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control_exports", type=Path, required=True)
    ap.add_argument("--persona_exports", type=Path, required=True)
    ap.add_argument("--round", type=int, default=5)
    ap.add_argument("--out_csv", type=Path, required=True)
    args = ap.parse_args()

    c_surv = _read_survival(args.control_exports / "survival_curve.csv")
    p_surv = _read_survival(args.persona_exports / "survival_curve.csv")
    c_tof = _read_tof(args.control_exports / "turn_of_failure.csv")
    p_tof = _read_tof(args.persona_exports / "turn_of_failure.csv")

    c_surv_r = _survival_at_r(c_surv, args.round)
    p_surv_r = _survival_at_r(p_surv, args.round)

    c_tof_agg = _aggregate_tof(c_tof)
    p_tof_agg = _aggregate_tof(p_tof)

    # control is typically a single persona label
    c_personas = sorted({r.persona for r in c_surv})
    if len(c_personas) != 1:
        raise ValueError(f"Expected 1 control persona in survival_curve.csv, got {c_personas}")
    c_key = c_personas[0]

    p_personas = _persona_keys(p_tof)

    # weighted persona aggregate (by total)
    def weighted(label: str) -> float:
        num = 0
        den = 0
        for pk in p_personas:
            c, t = p_tof_agg.get((pk, label), (0, 0))
            num += c
            den += t
        return _rate(num, den)

    # unweighted persona aggregate (mean of per-persona rates)
    def unweighted(label: str) -> float:
        rates: List[float] = []
        for pk in p_personas:
            c, t = p_tof_agg.get((pk, label), (0, 0))
            rates.append(_rate(c, t))
        return sum(rates) / len(rates) if rates else 0.0

    def control(label: str) -> float:
        c, t = c_tof_agg.get((c_key, label), (0, 0))
        return _rate(c, t)

    def control_surv() -> float:
        return float(c_surv_r.get(c_key, 0.0))

    def persona_surv_weighted() -> float:
        # weight by total at round r
        num = 0
        den = 0
        for pk in p_personas:
            # find survival row for pk at r
            sr = p_surv_r.get(pk)
            if sr is None:
                continue
            # recover survived/total from raw rows
            for row in p_surv:
                if row.persona == pk and row.round == args.round:
                    num += row.survived
                    den += row.total
                    break
        return _rate(num, den)

    def persona_surv_unweighted() -> float:
        vals = [float(p_surv_r[pk]) for pk in p_personas if pk in p_surv_r]
        return sum(vals) / len(vals) if vals else 0.0

    out_rows = [
        {
            "metric": f"Survival@{args.round}",
            "control": f"{control_surv():.2f}",
            "persona_weighted": f"{persona_surv_weighted():.2f}",
            "persona_unweighted": f"{persona_surv_unweighted():.2f}",
        },
        {
            "metric": "Fail@1",
            "control": f"{control('fail_at_1'):.2f}",
            "persona_weighted": f"{weighted('fail_at_1'):.2f}",
            "persona_unweighted": f"{unweighted('fail_at_1'):.2f}",
        },
        {
            "metric": "Never-fail",
            "control": f"{control('never_failed'):.2f}",
            "persona_weighted": f"{weighted('never_failed'):.2f}",
            "persona_unweighted": f"{unweighted('never_failed'):.2f}",
        },
    ]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        for r in out_rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
