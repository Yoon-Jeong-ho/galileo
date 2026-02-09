#!/usr/bin/env python3
"""Generate simple submission-ready SVG figures from tracked CSV artifacts.

Why stdlib-only?
- The writing environment may not have matplotlib installed.
- These figures are meant to be *vector* and easy to regenerate.

Input:
- docs/paper/artifacts/*.csv (tracked)

Output:
- docs/paper/figures/*.svg

Usage:
  python3 scripts/make_paper_figures_from_artifacts.py

Notes:
- Figure X (true survival curves) is generated from the aggregated artifact
  `survival_curve_personawise_control_vs_persona_seed1-4_mean_std_20260209.csv`.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "docs" / "paper" / "artifacts"
OUT = ROOT / "docs" / "paper" / "figures"
DATE_TAG = "20260209"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def write_svg(path: Path, svg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def svg_barh_deltas(
    items: List[Tuple[str, float]],
    *,
    title: str,
    xlabel: str,
    width: int = 1100,
    bar_h: int = 28,
    left_margin: int = 360,
    right_margin: int = 80,
    top_margin: int = 90,
    bottom_margin: int = 70,
    sort_by_abs: bool = True,
) -> str:
    if sort_by_abs:
        items = sorted(items, key=lambda x: abs(x[1]), reverse=True)
    else:
        items = sorted(items, key=lambda x: x[0])

    n = len(items)
    height = top_margin + bottom_margin + n * bar_h

    # scale
    max_abs = max([abs(v) for _, v in items] + [1e-9])
    plot_w = width - left_margin - right_margin

    def x(v: float) -> float:
        # map [-max_abs, +max_abs] to [0, plot_w]
        return (v / (2 * max_abs) + 0.5) * plot_w

    x0 = x(0.0)

    # svg header
    style = (
        "<style>"
        ".title{font: 20px sans-serif; font-weight:600;}"
        ".label{font: 14px sans-serif;}"
        ".tick{font: 12px sans-serif; fill:#333;}"
        ".val{font: 12px monospace; fill:#111;}"
        "</style>"
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        style,
        f'<text class="title" x="{left_margin}" y="34">{_escape(title)}</text>',
        f'<text class="label" x="{left_margin}" y="{height-22}">{_escape(xlabel)} (Δ in percentage points)</text>',
    ]

    # axis line at 0
    parts.append(
        f'<line x1="{left_margin + x0:.2f}" y1="{top_margin-8}" x2="{left_margin + x0:.2f}" y2="{height-bottom_margin+8}" stroke="#000" stroke-width="1"/>'
    )

    # ticks (5 ticks)
    for t in [-max_abs, -max_abs / 2, 0.0, max_abs / 2, max_abs]:
        xt = left_margin + x(t)
        parts.append(
            f'<line x1="{xt:.2f}" y1="{height-bottom_margin+2}" x2="{xt:.2f}" y2="{height-bottom_margin+8}" stroke="#000" stroke-width="1"/>'
        )
        parts.append(
            f'<text class="tick" x="{xt:.2f}" y="{height-bottom_margin+26}" text-anchor="middle">{t:+.1f}</text>'
        )

    # bars
    for i, (lab, v) in enumerate(items):
        y = top_margin + i * bar_h
        # label
        parts.append(
            f'<text class="label" x="{left_margin-10}" y="{y + bar_h*0.70:.2f}" text-anchor="end">{_escape(lab)}</text>'
        )
        # bar rect
        xv = x(v)
        x_start = min(x0, xv)
        w = abs(xv - x0)
        color = "#d62728" if v > 0 else "#1f77b4"
        parts.append(
            f'<rect x="{left_margin + x_start:.2f}" y="{y + 6}" width="{w:.2f}" height="{bar_h-12}" fill="{color}" opacity="0.9"/>'
        )
        # value text
        anchor = "start" if v >= 0 else "end"
        x_text = left_margin + xv + (6 if v >= 0 else -6)
        parts.append(
            f'<text class="val" x="{x_text:.2f}" y="{y + bar_h*0.70:.2f}" text-anchor="{anchor}">{v:+.2f}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def make_fig_table_w_effect_delta() -> None:
    p = ART / f"table_w_effect_delta_seed1-4_{DATE_TAG}.csv"
    rows = read_csv(p)
    items = [(r["metric"], float(r["delta_persona_minus_control_mean"])) for r in rows]
    svg = svg_barh_deltas(
        items,
        title="Table W effect sizes (persona pressure − neutral control) — seed1–4",
        xlabel="Effect size",
        sort_by_abs=True,
    )
    write_svg(OUT / f"table_w_effect_delta_seed1-4_{DATE_TAG}.svg", svg)


def make_fig_survival_r5_personawise() -> None:
    p = ART / f"survival_r5_personawise_control_vs_persona_seed1-4_mean_std_{DATE_TAG}.csv"
    rows = read_csv(p)
    items = [(r["persona"], float(r["delta_persona_minus_control_mean"])) for r in rows]
    svg = svg_barh_deltas(
        items,
        title="Persona-wise ΔSurvival@5 (persona pressure − control) — seed1–4",
        xlabel="ΔSurvival@5",
        sort_by_abs=True,
    )
    write_svg(OUT / f"survival_r5_personawise_delta_seed1-4_{DATE_TAG}.svg", svg)


def make_fig_tof_fail1_personawise() -> None:
    p = ART / f"tof_personawise_fail1_never_control_vs_persona_seed1-4_mean_std_{DATE_TAG}.csv"
    rows = [r for r in read_csv(p) if r.get("metric") == "Fail@1"]
    items = [(r["persona"], float(r["delta_persona_minus_control_mean"])) for r in rows]
    svg = svg_barh_deltas(
        items,
        title="Persona-wise ΔFail@1 (persona pressure − control) — seed1–4",
        xlabel="ΔFail@1",
        sort_by_abs=True,
    )
    write_svg(OUT / f"tof_personawise_fail1_delta_seed1-4_{DATE_TAG}.svg", svg)


def make_fig_recovery_personawise() -> None:
    p = ART / f"recovery_personawise_control_vs_persona_seed1-4_mean_std_{DATE_TAG}.csv"
    rows = read_csv(p)
    items = [(r["persona"], float(r["delta_persona_minus_control_mean"])) for r in rows]
    svg = svg_barh_deltas(
        items,
        title="Persona-wise ΔRecovery@flip (persona pressure − control) — seed1–4",
        xlabel="ΔRecovery@flip",
        sort_by_abs=True,
    )
    write_svg(OUT / f"recovery_personawise_delta_seed1-4_{DATE_TAG}.svg", svg)


def svg_line_chart(
    series: List[Tuple[str, List[Tuple[float, float]], str, str]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    width: int = 1100,
    height: int = 520,
    left_margin: int = 90,
    right_margin: int = 260,
    top_margin: int = 70,
    bottom_margin: int = 70,
) -> str:
    # series: (name, [(x,y)...], color, dasharray)
    xs = [x for _, pts, _, _ in series for x, _y in pts]
    ys = [y for _, pts, _, _ in series for _x, y in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = 0.0, max(ys)  # survival is percent
    plot_w = width - left_margin - right_margin
    plot_h = height - top_margin - bottom_margin

    def sx(x: float) -> float:
        return left_margin + (x - x_min) / (x_max - x_min) * plot_w if x_max != x_min else left_margin

    def sy(y: float) -> float:
        return top_margin + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h if y_max != y_min else top_margin + plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>'
        '.title{font: 20px sans-serif; font-weight:600;}'
        '.label{font: 14px sans-serif;}'
        '.tick{font: 12px sans-serif; fill:#333;}'
        '.legend{font: 13px sans-serif;}'
        '</style>',
        f'<text class="title" x="{left_margin}" y="34">{_escape(title)}</text>',
        f'<text class="label" x="{left_margin + plot_w/2:.2f}" y="{height-22}" text-anchor="middle">{_escape(xlabel)}</text>',
        f'<text class="label" x="22" y="{top_margin + plot_h/2:.2f}" transform="rotate(-90 22 {top_margin + plot_h/2:.2f})" text-anchor="middle">{_escape(ylabel)}</text>',
    ]

    # axes
    parts.append(f'<line x1="{left_margin}" y1="{top_margin+plot_h}" x2="{left_margin+plot_w}" y2="{top_margin+plot_h}" stroke="#000"/>')
    parts.append(f'<line x1="{left_margin}" y1="{top_margin}" x2="{left_margin}" y2="{top_margin+plot_h}" stroke="#000"/>')

    # x ticks (rounds)
    for x in range(int(x_min), int(x_max) + 1):
        xt = sx(float(x))
        parts.append(f'<line x1="{xt:.2f}" y1="{top_margin+plot_h}" x2="{xt:.2f}" y2="{top_margin+plot_h+6}" stroke="#000"/>')
        parts.append(f'<text class="tick" x="{xt:.2f}" y="{top_margin+plot_h+24}" text-anchor="middle">{x}</text>')

    # y ticks
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y_min + frac * (y_max - y_min)
        yt = sy(y)
        parts.append(f'<line x1="{left_margin-6}" y1="{yt:.2f}" x2="{left_margin}" y2="{yt:.2f}" stroke="#000"/>')
        parts.append(f'<text class="tick" x="{left_margin-10}" y="{yt+4:.2f}" text-anchor="end">{y:.0f}</text>')
        parts.append(f'<line x1="{left_margin}" y1="{yt:.2f}" x2="{left_margin+plot_w}" y2="{yt:.2f}" stroke="#bbb" stroke-width="1" opacity="0.6"/>')

    # lines
    for name, pts, color, dash in series:
        d = "M " + " L ".join([f"{sx(x):.2f} {sy(y):.2f}" for x, y in pts])
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"{dash_attr}/>')

    # legend
    lx = left_margin + plot_w + 20
    ly = top_margin + 10
    for i, (name, _pts, color, dash) in enumerate(series):
        y = ly + i * 22
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{lx}" y1="{y}" x2="{lx+30}" y2="{y}" stroke="{color}" stroke-width="3"{dash_attr}/>' )
        parts.append(f'<text class="legend" x="{lx+40}" y="{y+4}" >{_escape(name)}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def make_fig_survival_curves_rounds() -> None:
    p = ART / f"survival_curve_personawise_control_vs_persona_seed1-4_mean_std_{DATE_TAG}.csv"
    rows = read_csv(p)
    # pick personas: top 3 |delta| at r=5 + neutral control
    r5 = [r for r in rows if r["round"] == "5"]
    r5.sort(key=lambda r: abs(float(r["delta_persona_minus_control_mean"])), reverse=True)
    picked = [r["persona"] for r in r5[:3]]
    if "neutral_reask_control" not in picked:
        picked.append("neutral_reask_control")

    # build series for each persona: control mean and persona_pressure mean
    by_persona = {}
    for persona in picked:
        pr = [r for r in rows if r["persona"] == persona]
        pr.sort(key=lambda r: int(r["round"]))
        by_persona[persona] = pr

    colors = {
        "neutral_reask_control": "#000000",
        picked[0]: "#1f77b4",
        picked[1]: "#ff7f0e",
        picked[2]: "#2ca02c",
    }

    series = []
    # baseline: neutral control as dashed, and its pressure counterpart as dotted (if present)
    for persona in picked:
        pr = by_persona[persona]
        ctrl = [(float(r["round"]), float(r["control_mean"])) for r in pr]
        pres = [(float(r["round"]), float(r["persona_pressure_mean"])) for r in pr]
        c = colors.get(persona, "#9467bd")
        if persona == "neutral_reask_control":
            series.append((f"{persona} (control)", ctrl, c, "6,4"))
            series.append((f"{persona} (pressure)", pres, c, "2,4"))
        else:
            series.append((f"{persona} (control)", ctrl, c, ""))
            series.append((f"{persona} (pressure)", pres, c, "6,3"))

    svg = svg_line_chart(
        series,
        title="Survival curves over rounds (selected personas; seed1–4)",
        xlabel="Round",
        ylabel="Survival (%)",
    )
    write_svg(OUT / f"survival_curves_rounds_seed1-4_{DATE_TAG}.svg", svg)


def main() -> None:
    required = [
        ART / f"table_w_effect_delta_seed1-4_{DATE_TAG}.csv",
        ART / f"survival_r5_personawise_control_vs_persona_seed1-4_mean_std_{DATE_TAG}.csv",
        ART / f"tof_personawise_fail1_never_control_vs_persona_seed1-4_mean_std_{DATE_TAG}.csv",
        ART / f"recovery_personawise_control_vs_persona_seed1-4_mean_std_{DATE_TAG}.csv",
        ART / f"survival_curve_personawise_control_vs_persona_seed1-4_mean_std_{DATE_TAG}.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Missing required artifacts:\n" + "\n".join(missing))

    make_fig_table_w_effect_delta()
    make_fig_survival_r5_personawise()
    make_fig_survival_curves_rounds()
    make_fig_tof_fail1_personawise()
    make_fig_recovery_personawise()

    print(f"[OK] wrote SVG figures to {OUT}")


if __name__ == "__main__":
    main()
