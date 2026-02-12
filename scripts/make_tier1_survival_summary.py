#!/usr/bin/env python3
"""Generate Tier-1 cross-family survival summary CSVs (stdlib-only).

This script produces the small family-wise summaries used by:
- scripts/make_cross_family_figure_svg.py

It aggregates per-seed *paper export bundles* (paper_exports/) into a single CSV:
  model,persona,seeds,
  survival_r{R}_mean,survival_r{R}_std,
  delta_survival_r{R}_mean,delta_survival_r{R}_std,
  delta_fail_r1_mean,delta_fail_r1_std

Definitions (consistent with paper terminology):
- survival_rR: P(correct at every round 1..R), derived from paper_exports/survival_curve.csv
  (which is already a cumulative survival rate).
- fail_r1 = 1 - survival_r1.
- deltas are persona - control, computed per-seed then mean/std across seeds.

Inputs:
- One or more run roots, each containing:
    paper_exports/survival_curve.csv
    paper_exports/runner_metadata.json  (must include model + seed)

Example:
  python3 scripts/make_tier1_survival_summary.py \
    --run_roots results_paper/tier1_llama3_3b_seed1_20260212_030426,results_paper/tier1_llama3_3b_seed2_20260212_042339 \
    --out_csv docs/paper/artifacts/tier1_llama3_3b_seed1-2_survival_summary_YYYYMMDD.csv

NOTE: This repo intentionally does not track results_paper/; sync it locally when needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def std(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(v)


@dataclass(frozen=True)
class SeedBundle:
    run_root: Path
    model: str
    seed: int
    # persona -> round -> survival_rate in [0,1]
    survival: dict[str, dict[int, float]]


def load_bundle(run_root: Path) -> SeedBundle:
    exp = run_root / "paper_exports"
    surv_csv = exp / "survival_curve.csv"
    runner = exp / "runner_metadata.json"

    if not surv_csv.exists():
        raise FileNotFoundError(f"missing {surv_csv}")
    if not runner.exists():
        raise FileNotFoundError(f"missing {runner}")

    rm = read_json(runner)
    model = rm.get("model")
    seed = rm.get("seed")
    if not model:
        raise ValueError(f"runner_metadata.json missing model: {runner}")
    if seed is None:
        raise ValueError(f"runner_metadata.json missing seed: {runner}")

    by_persona: dict[str, dict[int, float]] = {}
    for r in read_csv(surv_csv):
        persona = (r.get("persona") or "").strip()
        rnd = int(r["round"])
        # survival_rate is a percent string with 6 decimals in paper_export.py
        rate = float(r["survival_rate"]) / 100.0
        by_persona.setdefault(persona, {})[rnd] = rate

    return SeedBundle(run_root=run_root, model=str(model), seed=int(seed), survival=by_persona)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_roots",
        required=True,
        help="Comma-separated run roots (each must contain paper_exports/)",
    )
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--round", type=int, default=5, help="Round R for survival_rR (default: 5)")
    ap.add_argument(
        "--control_persona",
        default="neutral_reask_control",
        help="Persona id for the neutral drift baseline (default: neutral_reask_control)",
    )

    args = ap.parse_args()

    run_roots = [Path(s).expanduser() for s in args.run_roots.split(",") if s.strip()]
    bundles = [load_bundle(p) for p in run_roots]

    models = {b.model for b in bundles}
    if len(models) != 1:
        raise SystemExit(f"expected exactly 1 model across bundles, got: {sorted(models)}")
    model = next(iter(models))

    seeds = sorted({b.seed for b in bundles})
    if len(seeds) != len(bundles):
        # duplicate seed paths typically indicate accidental reuse.
        raise SystemExit(f"duplicate seeds detected across bundles: seeds={seeds}")

    # Union of personas across seeds.
    personas = set()
    for b in bundles:
        personas.update(b.survival.keys())

    if args.control_persona not in personas:
        raise SystemExit(
            f"control_persona={args.control_persona} not found in any survival_curve.csv; personas={sorted(personas)}"
        )

    def get_surv(b: SeedBundle, persona: str, rnd: int) -> float:
        try:
            return float(b.survival[persona][rnd])
        except Exception:
            raise SystemExit(f"missing survival for persona={persona} round={rnd} in {b.run_root}")

    # Precompute per-seed control rates.
    control_surv_rR = {b.seed: get_surv(b, args.control_persona, args.round) for b in bundles}
    control_surv_r1 = {b.seed: get_surv(b, args.control_persona, 1) for b in bundles}

    rows = []

    for persona in sorted(personas, key=lambda p: (p != args.control_persona, p)):
        surv_rR = []
        delta_surv_rR = []
        delta_fail_r1 = []

        for b in bundles:
            s_rR = get_surv(b, persona, args.round)
            s_r1 = get_surv(b, persona, 1)

            surv_rR.append(s_rR)
            delta_surv_rR.append(s_rR - control_surv_rR[b.seed])

            fail_r1 = 1.0 - s_r1
            ctrl_fail_r1 = 1.0 - control_surv_r1[b.seed]
            delta_fail_r1.append(fail_r1 - ctrl_fail_r1)

        rows.append(
            {
                "model": model,
                "persona": persona,
                "seeds": ",".join(str(s) for s in seeds),
                f"survival_r{args.round}_mean": f"{mean(surv_rR):.8f}",
                f"survival_r{args.round}_std": f"{std(surv_rR):.12f}",
                f"delta_survival_r{args.round}_mean": f"{mean(delta_surv_rR):.8f}",
                f"delta_survival_r{args.round}_std": f"{std(delta_surv_rR):.12f}",
                "delta_fail_r1_mean": f"{mean(delta_fail_r1):.8f}",
                "delta_fail_r1_std": f"{std(delta_fail_r1):.12f}",
            }
        )

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "persona",
        "seeds",
        f"survival_r{args.round}_mean",
        f"survival_r{args.round}_std",
        f"delta_survival_r{args.round}_mean",
        f"delta_survival_r{args.round}_std",
        "delta_fail_r1_mean",
        "delta_fail_r1_std",
    ]

    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[OK] wrote {out} ({len(rows)} personas; seeds={seeds})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
