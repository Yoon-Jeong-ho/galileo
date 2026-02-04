#!/usr/bin/env python3
"""Matplotlib-free SVG figure generator for paper drafts.

Generates:
- survival_curve_<dataset>.svg : model-wise (7b/14b) persona-avg survival curve (r1..r5)
- fail1_never_<model>.svg : persona-wise bar chart for never vs fail@1 (aggregated over datasets)

Inputs:
- multi-seed results root:
    results_root/seed_k/<model>/adversarial_survival.csv
    results_root/seed_k/<model>/paper_exports/turn_of_failure.csv

No third-party dependencies.
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

PERSONA_ORDER = [
    "Authority Claim",
    "Strong Pressure",
    "Simple Denial",
    "Logical Trap",
    "Soft Pressure",
]
ROUNDS = [1, 2, 3, 4, 5]


def read_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_std(vals):
    vals = [float(v) for v in vals]
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def svg_header(w: int, h: int) -> str:
    return (
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\">\n"
        "<style>\n"
        "  .axis { stroke: #111; stroke-width: 1; }\n"
        "  .grid { stroke: #ddd; stroke-width: 1; }\n"
        "  .lbl { font: 12px sans-serif; fill: #111; }\n"
        "  .title { font: 14px sans-serif; font-weight: 600; fill: #111; }\n"
        "  .legend { font: 12px sans-serif; fill: #111; }\n"
        "</style>\n"
    )


def svg_footer() -> str:
    return "</svg>\n"


def line_chart_svg(title: str, xs, series, out_path: Path, y_min=0.0, y_max=100.0):
    """series: list[(name, ys, color)]"""
    w, h = 780, 420
    m = {"l": 60, "r": 20, "t": 40, "b": 50}
    pw = w - m["l"] - m["r"]
    ph = h - m["t"] - m["b"]

    def x_map(i: int) -> float:
        return m["l"] + (i / (len(xs) - 1)) * pw if len(xs) > 1 else float(m["l"])

    def y_map(y: float) -> float:
        return m["t"] + (1 - (y - y_min) / (y_max - y_min)) * ph

    parts = [svg_header(w, h)]
    parts.append(f"<text class=\"title\" x=\"{m[l]}\" y=\"22\">{title}</text>\n")

    # y grid
    for t in range(0, 101, 20):
        yy = y_map(float(t))
        parts.append(f"<line class=\"grid\" x1=\"{m[l]}\" y1=\"{yy:.1f}\" x2=\"{m[l]+pw}\" y2=\"{yy:.1f}\"/>\n")
        parts.append(f"<text class=\"lbl\" x=\"10\" y=\"{yy+4:.1f}\">{t}</text>\n")

    # axes
    parts.append(f"<line class=\"axis\" x1=\"{m[l]}\" y1=\"{m[t]}\" x2=\"{m[l]}\" y2=\"{m[t]+ph}\"/>\n")
    parts.append(f"<line class=\"axis\" x1=\"{m[l]}\" y1=\"{m[t]+ph}\" x2=\"{m[l]+pw}\" y2=\"{m[t]+ph}\"/>\n")

    # x ticks
    for i, x in enumerate(xs):
        xx = x_map(i)
        parts.append(f"<line class=\"grid\" x1=\"{xx:.1f}\" y1=\"{m[t]}\" x2=\"{xx:.1f}\" y2=\"{m[t]+ph}\"/>\n")
        parts.append(f"<text class=\"lbl\" x=\"{xx-8:.1f}\" y=\"{m[t]+ph+25}\">r{x}</text>\n")

    # lines
    for name, ys, color in series:
        pts = " ".join(f"{x_map(i):.1f},{y_map(y):.1f}" for i, y in enumerate(ys))
        parts.append(f"<polyline fill=\"none\" stroke=\"{color}\" stroke-width=\"2\" points=\"{pts}\"/>\n")
        for i, y in enumerate(ys):
            parts.append(f"<circle cx=\"{x_map(i):.1f}\" cy=\"{y_map(y):.1f}\" r=\"3\" fill=\"{color}\"/>\n")

    # legend
    lx = m["l"] + pw - 220
    ly = m["t"] + 10
    for i, (name, _, color) in enumerate(series):
        parts.append(f"<rect x=\"{lx}\" y=\"{ly+i*18}\" width=\"10\" height=\"10\" fill=\"{color}\"/>\n")
        parts.append(f"<text class=\"legend\" x=\"{lx+14}\" y=\"{ly+10+i*18}\">{name}</text>\n")

    parts.append(svg_footer())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts), encoding="utf-8")


def bar_chart_svg(title: str, personas, series, out_path: Path, y_max=100.0):
    """series: list[(name, values_per_persona, color)]"""
    w, h = 780, 420
    m = {"l": 80, "r": 20, "t": 40, "b": 140}
    pw = w - m["l"] - m["r"]
    ph = h - m["t"] - m["b"]

    def y_map(y: float) -> float:
        return m["t"] + (1 - y / y_max) * ph

    parts = [svg_header(w, h)]
    parts.append(f"<text class=\"title\" x=\"{m[l]}\" y=\"22\">{title}</text>\n")

    for t in range(0, 101, 20):
        yy = y_map(float(t))
        parts.append(f"<line class=\"grid\" x1=\"{m[l]}\" y1=\"{yy:.1f}\" x2=\"{m[l]+pw}\" y2=\"{yy:.1f}\"/>\n")
        parts.append(f"<text class=\"lbl\" x=\"10\" y=\"{yy+4:.1f}\">{t}</text>\n")

    parts.append(f"<line class=\"axis\" x1=\"{m[l]}\" y1=\"{m[t]}\" x2=\"{m[l]}\" y2=\"{m[t]+ph}\"/>\n")
    parts.append(f"<line class=\"axis\" x1=\"{m[l]}\" y1=\"{m[t]+ph}\" x2=\"{m[l]+pw}\" y2=\"{m[t]+ph}\"/>\n")

    n = len(personas)
    g_w = pw / n
    bar_w = min(22.0, (g_w - 10.0) / max(1, len(series)))

    for i, persona in enumerate(personas):
        gx = m["l"] + i * g_w
        # rotated label
        parts.append(
            f"<text class=\"lbl\" x=\"{gx+5:.1f}\" y=\"{m[t]+ph+35}\" "
            f"transform=\"rotate(45 {gx+5:.1f},{m[t]+ph+35}\"\">{persona}</text>\n"
        )
        for j, (name, vals, color) in enumerate(series):
            v = float(vals[i])
            x = gx + 5 + j * (bar_w + 4)
            y = y_map(v)
            height = (m["t"] + ph) - y
            parts.append(f"<rect x=\"{x:.1f}\" y=\"{y:.1f}\" width=\"{bar_w:.1f}\" height=\"{height:.1f}\" fill=\"{color}\"/>\n")

    lx = m["l"] + pw - 240
    ly = m["t"] + 10
    for i, (name, _, color) in enumerate(series):
        parts.append(f"<rect x=\"{lx}\" y=\"{ly+i*18}\" width=\"10\" height=\"10\" fill=\"{color}\"/>\n")
        parts.append(f"<text class=\"legend\" x=\"{lx+14}\" y=\"{ly+10+i*18}\">{name}</text>\n")

    parts.append(svg_footer())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--models", default="7b,14b")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    # persona-avg survival curve per dataset per model (mean over seeds)
    ds_round = defaultdict(list)  # (model, ds, rd) -> [persona-avg rate per seed]

    for seed_dir in sorted(results_root.glob("seed_*")):
        for model in models:
            base = seed_dir / model
            adv_path = base / "adversarial_survival.csv"
            if not adv_path.exists():
                continue
            rows = read_csv(adv_path)
            tmp = defaultdict(lambda: [0, 0])
            for r in rows:
                key = (r["test_name"], r["persona"], int(r["round"]))
                tmp[key][0] += int(r["survived"])
                tmp[key][1] += int(r["total"])

            datasets = sorted({r["test_name"] for r in rows})
            for ds in datasets:
                for rd in ROUNDS:
                    vals = []
                    for p in PERSONA_ORDER:
                        s, t = tmp.get((ds, p, rd), (0, 0))
                        if t:
                            vals.append(s / t * 100.0)
                    if vals:
                        ds_round[(model, ds, rd)].append(sum(vals) / len(vals))

    colors = {"7b": "#1f77b4", "14b": "#d62728"}

    for ds in sorted({k[1] for k in ds_round.keys()}):
        series = []
        for model in models:
            ys = []
            ok = True
            for rd in ROUNDS:
                key = (model, ds, rd)
                if key not in ds_round:
                    ok = False
                    break
                m, _ = mean_std(ds_round[key])
                ys.append(m)
            if ok:
                series.append((model, ys, colors.get(model, "#333")))
        if series:
            line_chart_svg(
                title=f"Survival curve (persona-avg): {ds}",
                xs=ROUNDS,
                series=series,
                out_path=out_dir / f"survival_curve_{ds}.svg",
            )

    # never vs fail@1 (aggregate over datasets) per model
    for model in models:
        never = defaultdict(list)
        fail1 = defaultdict(list)
        for seed_dir in sorted(results_root.glob("seed_*")):
            tof_path = seed_dir / model / "paper_exports" / "turn_of_failure.csv"
            if not tof_path.exists():
                continue
            rows = read_csv(tof_path)
            by = defaultdict(lambda: defaultdict(int))
            tot = defaultdict(int)
            for r in rows:
                persona = r["persona"]
                ft = int(r["fail_turn"])
                cnt = int(r["count"])
                by[persona][ft] += cnt
                tot[persona] += cnt
            for persona in PERSONA_ORDER:
                total = tot.get(persona, 0)
                if not total:
                    continue
                never[persona].append(by[persona].get(0, 0) / total * 100.0)
                fail1[persona].append(by[persona].get(1, 0) / total * 100.0)

        if never:
            never_mean = [mean_std(never[p])[0] if never.get(p) else 0.0 for p in PERSONA_ORDER]
            fail1_mean = [mean_std(fail1[p])[0] if fail1.get(p) else 0.0 for p in PERSONA_ORDER]
            bar_chart_svg(
                title=f"Turn-of-failure summary (aggregate): {model}",
                personas=PERSONA_ORDER,
                series=[("never", never_mean, "#2ca02c"), ("fail@1", fail1_mean, "#ff7f0e")],
                out_path=out_dir / f"fail1_never_{model}.svg",
            )


if __name__ == "__main__":
    main()
