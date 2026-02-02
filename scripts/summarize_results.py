#!/usr/bin/env python3
"""Summarize Galileo CSV outputs (stdlib only).

Reads initial_accuracy.csv / adversarial_survival.csv / recovery_accuracy.csv
and prints a short markdown summary.

Usage:
  python scripts/summarize_results.py --results_root /mnt/.../all_pilot_xxx/7b
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True, help="Path that contains the CSVs (e.g., .../7b)")
    ap.add_argument("--round", type=int, default=5, help="Adversarial round to summarize")
    args = ap.parse_args()

    root = Path(args.results_root)
    init = read_csv(root / "initial_accuracy.csv")
    adv = read_csv(root / "adversarial_survival.csv")
    rec = read_csv(root / "recovery_accuracy.csv")

    model = init[0]["model"] if init else "(unknown)"

    # Initial accuracy per dataset
    init_rows = sorted(
        ((r["test_name"], float(r["accuracy"])) for r in init),
        key=lambda x: (-x[1], x[0]),
    )

    # Survival at a target round (aggregate across datasets by summing counts)
    surv = defaultdict(lambda: {"survived": 0, "total": 0})
    surv_by_ds = defaultdict(lambda: defaultdict(lambda: {"survived": 0, "total": 0}))
    for r in adv:
        if int(r["round"]) != args.round:
            continue
        p = r["persona"]
        surv[p]["survived"] += int(r["survived"])
        surv[p]["total"] += int(r["total"])
        ds = r["test_name"]
        surv_by_ds[ds][p]["survived"] += int(r["survived"])
        surv_by_ds[ds][p]["total"] += int(r["total"])

    surv_rows = []
    for p, c in surv.items():
        tot = c["total"]
        rate = (c["survived"] / tot * 100.0) if tot else 0.0
        surv_rows.append((p, rate, c["survived"], tot))
    surv_rows.sort(key=lambda x: x[1])  # worst to best

    # Recovery (aggregate across datasets)
    rec_agg = defaultdict(lambda: {"recovered": 0, "total": 0})
    for r in rec:
        p = r["persona"]
        rec_agg[p]["recovered"] += int(r["recovered"])
        rec_agg[p]["total"] += int(r["total"])

    rec_rows = []
    for p, c in rec_agg.items():
        tot = c["total"]
        rate = (c["recovered"] / tot * 100.0) if tot else 0.0
        rec_rows.append((p, rate, c["recovered"], tot))
    rec_rows.sort(key=lambda x: x[1])

    print(f"### Results snapshot ({model})")
    print("")
    print(f"- Results dir: `{root}`")
    print("")

    print("**Initial accuracy (per dataset)**")
    for ds, acc in init_rows:
        print(f"- {ds}: {acc:.2f}%")
    print("")

    print(f"**Adversarial survival @ round {args.round} (aggregated over datasets; lower = more vulnerable)**")
    for p, rate, s, t in surv_rows:
        print(f"- {p}: {rate:.2f}% ({s}/{t})")
    print("")

    print("**Recovery rate (after a flip; aggregated over datasets)**")
    for p, rate, s, t in rec_rows:
        print(f"- {p}: {rate:.2f}% ({s}/{t})")


if __name__ == "__main__":
    main()
