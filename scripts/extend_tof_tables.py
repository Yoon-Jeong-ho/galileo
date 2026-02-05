#!/usr/bin/env python3
"""Extend PAPER_RESULTS_ANALYSIS_KO.md with dataset×persona turn-of-failure stats.

Uses per-seed paper_exports/turn_of_failure.csv produced by paper_export.py.
Outputs mean±std for never (fail_turn=0) and fail@1 (fail_turn=1).

Stdlib only.
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

PERSONAS = ["Authority Claim","Strong Pressure","Simple Denial","Logical Trap","Soft Pressure"]


def mean_std(vals):
    vals = [float(v) for v in vals]
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def read_rows(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--models", default="7b,14b")
    ap.add_argument("--start_tag", default="<!-- AUTO:TOF_DATASET_TABLE_START -->")
    ap.add_argument("--end_tag", default="<!-- AUTO:TOF_DATASET_TABLE_END -->")
    args = ap.parse_args()

    root = Path(args.results_root)
    out_md = Path(args.out_md)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    seeds = sorted([p.name for p in root.glob("seed_*" ) if p.is_dir()])

    # (model, test_name, persona, ft) -> [rate per seed]
    rates = defaultdict(list)

    for seed in seeds:
        for model_tag in models:
            p = root / seed / model_tag / "paper_exports" / "turn_of_failure.csv"
            if not p.exists():
                continue
            rows = read_rows(p)
            # (test, persona) total and per-ft
            total = defaultdict(int)
            per = defaultdict(lambda: defaultdict(int))
            for r in rows:
                test = r["test_name"]
                persona = r["persona"]
                ft = int(r["fail_turn"])
                cnt = int(r["count"])
                total[(test, persona)] += cnt
                per[(test, persona)][ft] += cnt
            for (test, persona), tot in total.items():
                if tot <= 0:
                    continue
                for ft in (0, 1):
                    rates[(model_tag, test, persona, ft)].append(per[(test, persona)].get(ft, 0) / tot * 100.0)

    tests = sorted({k[1] for k in rates.keys()})

    def cell(model_tag, test, persona, ft):
        v = rates.get((model_tag, test, persona, ft), [])
        if not v:
            return "-"
        m, s = mean_std(v)
        return f"{m:.2f}±{s:.2f}"

    blocks = []
    blocks.append("\n\n" + args.start_tag + "\n")
    blocks.append("\n## 8.X Dataset×Persona Turn-of-failure (never / fail@1, mean±std over seeds)\n\n")
    blocks.append(f"Results root: `{root}`\n\n")

    for model_tag in models:
        blocks.append(f"### {model_tag.upper()}\n\n")
        for test in tests:
            blocks.append(f"#### {test}\n\n")
            blocks.append("| persona | never | fail@1 |\n")
            blocks.append("|---|---:|---:|\n")
            for persona in PERSONAS:
                never = cell(model_tag, test, persona, 0)
                f1 = cell(model_tag, test, persona, 1)
                blocks.append(f"| {persona} | {never} | {f1} |\n")
            blocks.append("\n")

    blocks.append(args.end_tag + "\n")

    text = out_md.read_text(encoding="utf-8")
    if args.start_tag in text and args.end_tag in text:
        pre = text.split(args.start_tag)[0].rstrip() + "\n\n"
        post = text.split(args.end_tag)[1].lstrip()
        out_md.write_text(pre + "".join(blocks) + "\n" + post, encoding="utf-8")
        print("replaced")
    else:
        out_md.write_text(text.rstrip() + "\n" + "".join(blocks), encoding="utf-8")
        print("appended")


if __name__ == "__main__":
    main()
