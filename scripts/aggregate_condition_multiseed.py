#!/usr/bin/env python3
"""Aggregate single-model condition runs laid out as <root>/<dataset_alias>/seed_<n>/.

Outputs:
- metrics_by_seed.csv
- metrics_mean_std.csv
- delta_mean_std.csv
- summary.md
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def persona_key(name: str) -> str:
    low = (name or "").strip().lower()
    if low in {"control re-asking", "control_reask", "neutral_reask_control"}:
        return "control_reask"
    if low in {"authority claim", "authority_claim"}:
        return "authority_claim"
    return low


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--round", type=int, default=5)
    args = ap.parse_args()

    root = Path(args.results_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    by_seed_rows = []
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for seed_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")):
            if not (seed_dir / "initial_accuracy.csv").exists():
                continue
            if not (seed_dir / "adversarial_survival.csv").exists():
                continue
            if not (seed_dir / "paper_exports" / "turn_of_failure.csv").exists():
                continue
            initial_rows = read_csv(seed_dir / "initial_accuracy.csv")
            surv_rows = read_csv(seed_dir / "adversarial_survival.csv")
            rec_rows = read_csv(seed_dir / "recovery_accuracy.csv") if (seed_dir / "recovery_accuracy.csv").exists() else []

            dataset_name = initial_rows[0]["test_name"] if initial_rows else dataset_dir.name
            initial_acc = float(initial_rows[0]["accuracy"]) / 100.0 if initial_rows else 0.0

            arm = {}
            for row in surv_rows:
                if int(row["round"]) != args.round:
                    continue
                arm[persona_key(row["persona"])] = {
                    "survival_r5": float(row["survival_rate"]) / 100.0,
                    "total": int(row["total"]),
                    "survived": int(row["survived"]),
                }
            fail1 = defaultdict(float)
            first_turn_rows = [r for r in read_csv(seed_dir / "Qwen2.5-7B-Instruct" / f"{dataset_name}_adversarial.jsonl".replace(".jsonl", ""))] if False else None
            # Fail@1 from paper export CSV is easier.
            tof = read_csv(seed_dir / "paper_exports" / "turn_of_failure.csv")
            for row in tof:
                if row["test_name"] != dataset_name:
                    continue
                if row["fail_turn"] == "1":
                    fail1[persona_key(row["persona"])] += float(row["rate"]) / 100.0

            rec = {}
            for row in rec_rows:
                rec[persona_key(row["persona"])] = {
                    "recovery_rate": float(row["recovery_rate"]) / 100.0,
                    "recovery_denominator": int(row["total"]),
                    "recovered": int(row["recovered"]),
                }

            for persona in sorted(arm):
                post_recovery = (
                    (arm[persona]["survived"] + rec.get(persona, {}).get("recovered", 0)) / arm[persona]["total"]
                    if arm[persona]["total"] else 0.0
                )
                by_seed_rows.append(
                    {
                        "dataset_alias": dataset_dir.name,
                        "dataset_name": dataset_name,
                        "seed": seed_dir.name.replace("seed_", ""),
                        "persona": persona,
                        "initial_accuracy": f"{initial_acc:.6f}",
                        "survival_r5": f"{arm[persona]['survival_r5']:.6f}",
                        "fail1": f"{fail1.get(persona, 0.0):.6f}",
                        "recovery_rate": f"{rec.get(persona, {}).get('recovery_rate', 0.0):.6f}",
                        "recovery_denominator": rec.get(persona, {}).get("recovery_denominator", 0),
                        "post_recovery_acc": f"{post_recovery:.6f}",
                    }
                )

    write_csv(
        out_dir / "metrics_by_seed.csv",
        [
            "dataset_alias",
            "dataset_name",
            "seed",
            "persona",
            "initial_accuracy",
            "survival_r5",
            "fail1",
            "recovery_rate",
            "recovery_denominator",
            "post_recovery_acc",
        ],
        by_seed_rows,
    )

    grouped = defaultdict(list)
    for row in by_seed_rows:
        grouped[(row["dataset_alias"], row["dataset_name"], row["persona"])].append(row)

    mean_rows = []
    for key, rows in sorted(grouped.items()):
        dataset_alias, dataset_name, persona = key
        for metric in ["initial_accuracy", "survival_r5", "fail1", "recovery_rate", "post_recovery_acc"]:
            vals = [float(r[metric]) for r in rows]
            m, sd = mean_std(vals)
            mean_rows.append(
                {
                    "dataset_alias": dataset_alias,
                    "dataset_name": dataset_name,
                    "persona": persona,
                    "metric": metric,
                    "mean": f"{m:.6f}",
                    "std": f"{sd:.6f}",
                    "n_seeds": len(vals),
                }
            )
    write_csv(
        out_dir / "metrics_mean_std.csv",
        ["dataset_alias", "dataset_name", "persona", "metric", "mean", "std", "n_seeds"],
        mean_rows,
    )

    delta_rows = []
    by_seed_dataset = defaultdict(dict)
    for row in by_seed_rows:
        by_seed_dataset[(row["dataset_alias"], row["dataset_name"], row["seed"])][row["persona"]] = row
    per_dataset_metric = defaultdict(list)
    for (dataset_alias, dataset_name, seed), personas in sorted(by_seed_dataset.items()):
        if "control_reask" not in personas or "authority_claim" not in personas:
            continue
        ctl = personas["control_reask"]
        auth = personas["authority_claim"]
        for metric in ["survival_r5", "fail1", "post_recovery_acc"]:
            delta = float(auth[metric]) - float(ctl[metric])
            per_dataset_metric[(dataset_alias, dataset_name, metric)].append(delta)
    for key, vals in sorted(per_dataset_metric.items()):
        dataset_alias, dataset_name, metric = key
        m, sd = mean_std(vals)
        delta_rows.append(
            {
                "dataset_alias": dataset_alias,
                "dataset_name": dataset_name,
                "metric": metric,
                "mean_delta_authority_minus_control": f"{m:.6f}",
                "std_delta_authority_minus_control": f"{sd:.6f}",
                "n_seeds": len(vals),
            }
        )
    write_csv(
        out_dir / "delta_mean_std.csv",
        ["dataset_alias", "dataset_name", "metric", "mean_delta_authority_minus_control", "std_delta_authority_minus_control", "n_seeds"],
        delta_rows,
    )

    md = [f"# Multiseed summary for {root.name}\n\n"]
    for row in delta_rows:
        md.append(
            f"- {row['dataset_name']} / {row['metric']}: "
            f"{row['mean_delta_authority_minus_control']} ± {row['std_delta_authority_minus_control']} (n={row['n_seeds']})\n"
        )
    (out_dir / "summary.md").write_text("".join(md), encoding="utf-8")

    print(f"Wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
