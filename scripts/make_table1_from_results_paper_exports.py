#!/usr/bin/env python3
"""Build Table-1-ready per-family metrics directly from results_paper/*/paper_exports.

Motivation
- Our current Tier-1 cross-family artifacts (tier1_*_survival_summary_*.csv) often
  lack absolute Fail@1 and Recovery exports, which makes the main table look
  “missing results”.
- However, every paper-ready run under results_paper/<alias>/paper_exports has:
  - survival_curve.csv
  - turn_of_failure.csv

This script aggregates those exports into a single tracked CSV we can cite / use
for Table 1:
  - Survival@5 (NRC vs persona aggregate + delta)
  - Fail@1 (NRC vs persona aggregate + delta)

Aggregation
- For a given run alias, we compute per-persona Survival@5 and Fail@1 by micro-
  averaging across tasks (sum counts / sum totals).
- Persona aggregate: equal-weight mean across personas of (persona - NRC) deltas,
  then reconstruct Persona absolute as NRC + mean(delta).
  (Matches our current Table-1 semantics when denominators |C_p| are unavailable.)

Usage (run on nlp8 repo, then commit artifacts):
  python3 scripts/make_table1_from_results_paper_exports.py \
    --results_paper /data_x/aa007878/galileo/results_paper \
    --out docs/paper/artifacts/table1_from_results_paper_exports_$(date +%Y%m%d).csv

Notes
- Recovery@flip is NOT exported in paper_exports for most runs today, so this
  script does not attempt to fill it.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import json


NRC_ID = "neutral_reask_control"


@dataclass
class PersonaMetrics:
    surv5: float  # in [0,1]
    fail1: float  # in [0,1]


def _read_csv(p: Path) -> list[dict[str, str]]:
    with p.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _load_survival5_by_persona(survival_curve_csv: Path) -> dict[str, PersonaMetrics]:
    rows = _read_csv(survival_curve_csv)

    # survival_curve.csv schema (from paper_exports):
    # persona,round,survived,total,survival_rate
    # We want round==5 micro-avg across tasks: sum(survived)/sum(total)
    accum: dict[str, dict[str, int]] = {}
    for r in rows:
        if str(r["round"]).strip() != "5":
            continue
        persona = r["persona"].strip()
        survived = int(float(r["survived"]))
        total = int(float(r["total"]))
        a = accum.setdefault(persona, {"survived": 0, "total": 0})
        a["survived"] += survived
        a["total"] += total

    out: dict[str, PersonaMetrics] = {}
    for persona, a in accum.items():
        surv5 = (a["survived"] / a["total"]) if a["total"] else float("nan")
        out[persona] = PersonaMetrics(surv5=surv5, fail1=float("nan"))
    return out


def _load_fail1_by_persona(tof_csv: Path) -> dict[str, float]:
    rows = _read_csv(tof_csv)

    # turn_of_failure.csv schema (from paper_exports):
    # persona,test_name,fail_turn,fail_turn_label,count,total,rate
    # We want Fail@1 micro-avg across tasks: sum(count where fail_turn==1)/sum(total)
    numer: dict[str, int] = {}
    denom: dict[str, int] = {}

    for r in rows:
        persona = r["persona"].strip()
        fail_turn = int(float(r["fail_turn"]))
        cnt = int(float(r["count"]))
        tot = int(float(r["total"]))

        # Add denom once per row; this over-counts if multiple fail_turn entries per task.
        # Fix: denom should be per (persona,test_name) total, not per fail_turn row.
        # We'll track seen totals per (persona,test_name) and add once.
        # We still sum numerator only for fail_turn==1.

    # Second pass with proper denom accounting
    seen_totals: set[tuple[str, str]] = set()
    for r in rows:
        persona = r["persona"].strip()
        test = r["test_name"].strip()
        key = (persona, test)
        tot = int(float(r["total"]))
        if key not in seen_totals:
            denom[persona] = denom.get(persona, 0) + tot
            seen_totals.add(key)

        fail_turn = int(float(r["fail_turn"]))
        if fail_turn == 1:
            cnt = int(float(r["count"]))
            numer[persona] = numer.get(persona, 0) + cnt

    out: dict[str, float] = {}
    for persona in denom:
        out[persona] = (numer.get(persona, 0) / denom[persona]) if denom[persona] else float("nan")
    return out


def _infer_model_name(paper_exports_dir: Path) -> str:
    meta = paper_exports_dir / "metadata.json"
    if not meta.exists():
        return ""
    try:
        j = json.loads(meta.read_text(encoding="utf-8"))
        model_dir = str(j.get("model_dir", ""))
        if model_dir:
            return Path(model_dir).name
    except Exception:
        return ""
    return ""


def compute_run_metrics(paper_exports_dir: Path) -> dict[str, float]:
    surv_csv = paper_exports_dir / "survival_curve.csv"
    tof_csv = paper_exports_dir / "turn_of_failure.csv"

    surv = _load_survival5_by_persona(surv_csv)
    fail = _load_fail1_by_persona(tof_csv)

    if NRC_ID not in surv or NRC_ID not in fail:
        raise ValueError(f"Missing NRC persona '{NRC_ID}' in {paper_exports_dir}")

    nrc_surv5 = surv[NRC_ID].surv5
    nrc_fail1 = fail[NRC_ID]

    personas = sorted([p for p in surv.keys() if p != NRC_ID])
    if not personas:
        raise ValueError(f"No personas found (besides NRC) in {paper_exports_dir}")

    surv_deltas = [(surv[p].surv5 - nrc_surv5) for p in personas]
    fail_deltas = [(fail.get(p, float("nan")) - nrc_fail1) for p in personas if p in fail]

    # equal-weight mean of deltas
    delta_surv5 = _mean(surv_deltas)
    delta_fail1 = _mean(fail_deltas) if fail_deltas else float("nan")

    persona_surv5 = nrc_surv5 + delta_surv5
    persona_fail1 = nrc_fail1 + delta_fail1

    return {
        "model_name": _infer_model_name(paper_exports_dir),
        "nrc_survival_r5": nrc_surv5,
        "persona_survival_r5": persona_surv5,
        "delta_survival_r5": delta_surv5,
        "nrc_fail1": nrc_fail1,
        "persona_fail1": persona_fail1,
        "delta_fail1": delta_fail1,
        "num_personas": float(len(personas)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_paper", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--alias_to_model",
        type=Path,
        default=None,
        help="Optional CSV mapping alias->model_display_name (columns: alias,model).",
    )
    args = ap.parse_args()

    alias2model: dict[str, str] = {}
    if args.alias_to_model is not None:
        rows = _read_csv(args.alias_to_model)
        for r in rows:
            alias2model[r["alias"].strip()] = r["model"].strip()

    out_rows: list[dict[str, str]] = []
    for p in sorted(args.results_paper.glob("*/paper_exports")):
        alias = p.parent.name
        try:
            m = compute_run_metrics(p)
        except Exception as e:
            # Keep going; we want partial progress rather than failing the whole table.
            out_rows.append(
                {
                    "alias": alias,
                    "model": alias2model.get(alias, ""),
                    "status": f"FAIL:{type(e).__name__}:{str(e)[:120]}",
                }
            )
            continue

        out_rows.append(
            {
                "alias": alias,
                "model": alias2model.get(alias, "") or m.get("model_name", ""),
                "status": "OK",
                "nrc_survival_r5": f"{m['nrc_survival_r5']:.6f}",
                "persona_survival_r5": f"{m['persona_survival_r5']:.6f}",
                "delta_survival_r5": f"{m['delta_survival_r5']:.6f}",
                "nrc_fail1": f"{m['nrc_fail1']:.6f}",
                "persona_fail1": f"{m['persona_fail1']:.6f}",
                "delta_fail1": f"{m['delta_fail1']:.6f}",
                "num_personas": f"{m['num_personas']:.0f}",
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "alias",
        "model",
        "status",
        "nrc_survival_r5",
        "persona_survival_r5",
        "delta_survival_r5",
        "nrc_fail1",
        "persona_fail1",
        "delta_fail1",
        "num_personas",
    ]
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in cols})

    print(f"Wrote: {args.out} ({len(out_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
