#!/usr/bin/env python3
"""Aggregate multi-seed results into paper-ready tables (stdlib only)."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_std(xs):
    xs = [float(x) for x in xs]
    if not xs:
        return (0.0, 0.0)
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return (m, 0.0)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return (m, math.sqrt(var))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--round", type=int, default=5)
    args = ap.parse_args()

    root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = sorted([p for p in root.glob("seed_*") if p.is_dir()], key=lambda p: p.name)
    model_tags = sorted({d.name for s in seeds for d in s.iterdir() if d.is_dir()})

    init = defaultdict(list)  # (model, dataset) -> [acc]
    surv = defaultdict(list)  # (model, persona) -> [survival@R]
    rec = defaultdict(list)   # (model, persona) -> [recovery]

    for seed_dir in seeds:
        for tag in model_tags:
            mdir = seed_dir / tag
            if not mdir.exists():
                continue

            ip = mdir / "initial_accuracy.csv"
            if ip.exists():
                for r in read_csv(ip):
                    init[(tag, r["test_name"])].append(float(r["accuracy"]))

            apath = mdir / "adversarial_survival.csv"
            if apath.exists():
                rows = read_csv(apath)
                by_p = defaultdict(lambda: [0, 0])
                for r in rows:
                    if int(r["round"]) != args.round:
                        continue
                    p = r["persona"]
                    by_p[p][0] += int(r["survived"])
                    by_p[p][1] += int(r["total"])
                for p, (s, t) in by_p.items():
                    surv[(tag, p)].append((s / t * 100.0) if t else 0.0)

            rpath = mdir / "recovery_accuracy.csv"
            if rpath.exists():
                rows = read_csv(rpath)
                by_p = defaultdict(lambda: [0, 0])
                for r in rows:
                    p = r["persona"]
                    by_p[p][0] += int(r["recovered"])
                    by_p[p][1] += int(r["total"])
                for p, (s, t) in by_p.items():
                    rec[(tag, p)].append((s / t * 100.0) if t else 0.0)

    def write_table_csv(path: Path, header, rows):
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    init_rows = []
    for (tag, ds), vals in sorted(init.items(), key=lambda x: (x[0][0], x[0][1])):
        m, sd = mean_std(vals)
        init_rows.append([tag, ds, f"{m:.2f}", f"{sd:.2f}", str(len(vals))])
    write_table_csv(out_dir / "table_initial.csv", ["model", "test_name", "acc_mean", "acc_std", "n_seeds"], init_rows)

    surv_rows = []
    for (tag, persona), vals in sorted(surv.items(), key=lambda x: (x[0][0], x[0][1])):
        m, sd = mean_std(vals)
        surv_rows.append([tag, persona, f"{m:.2f}", f"{sd:.2f}", str(len(vals))])
    write_table_csv(out_dir / f"table_survival_r{args.round}.csv", ["model", "persona", "survival_mean", "survival_std", "n_seeds"], surv_rows)

    rec_rows = []
    for (tag, persona), vals in sorted(rec.items(), key=lambda x: (x[0][0], x[0][1])):
        m, sd = mean_std(vals)
        rec_rows.append([tag, persona, f"{m:.2f}", f"{sd:.2f}", str(len(vals))])
    write_table_csv(out_dir / "table_recovery.csv", ["model", "persona", "recovery_mean", "recovery_std", "n_seeds"], rec_rows)

    md = []
    md.append(f"# Multi-seed summary (round {args.round})\n")
    md.append(f"Results root: `{root}`\n")
    md.append("Seeds: " + ", ".join([s.name for s in seeds]) + "\n\n")

    md.append(f"## Survival @ round {args.round} (mean±std over seeds)\n")
    for tag in model_tags:
        md.append(f"### {tag}\n")
        per = []
        for (t, persona), vals in surv.items():
            if t != tag:
                continue
            m, sd = mean_std(vals)
            per.append((m, persona, sd, len(vals)))
        per.sort()
        for m, persona, sd, n in per:
            md.append(f"- {persona}: {m:.2f} ± {sd:.2f} (n={n})\n")
        md.append("\n")

    (out_dir / "table_summary.md").write_text("".join(md), encoding="utf-8")

    print("Wrote:")
    print(out_dir / "table_initial.csv")
    print(out_dir / f"table_survival_r{args.round}.csv")
    print(out_dir / "table_recovery.csv")
    print(out_dir / "table_summary.md")


if __name__ == "__main__":
    main()
