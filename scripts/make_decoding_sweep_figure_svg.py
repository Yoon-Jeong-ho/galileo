#!/usr/bin/env python3
"""Generate a compact SVG figure summarizing the decoding sensitivity sweep.

Stdlib-only (no matplotlib) for portability.

Input (tracked artifact):
- docs/paper/artifacts/decoding_sweep_qwen_temp_summary_seed1-2_20260211.csv

Output:
- docs/paper/figures/decoding_sweep_qwen_delta_seed1-2_20260211.svg

The figure plots, for temp in {0.0, 0.7}:
- ΔSurvival@5 (persona-mean − control)
- ΔFail@1 (persona-mean − control)

We use the mean across seed1–2 rows.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "docs" / "paper" / "artifacts"
OUT = ROOT / "docs" / "paper" / "figures"
DATE_TAG = "20260211"


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


def main() -> None:
    p = ART / f"decoding_sweep_qwen_temp_summary_seed1-2_{DATE_TAG}.csv"
    rows = read_csv(p)

    want = {}
    for r in rows:
        if r["seed"] != "mean_seed1-2":
            continue
        temp = r["temp"]
        want[temp] = (
            float(r["delta_survival_r5_persona_minus_control"]),
            float(r["delta_fail_r1_persona_minus_control"]),
        )

    # enforce ordering
    items: List[Tuple[str, float, float]] = [
        ("0.0",) + want["temp0"],
        ("0.7",) + want["temp0p7"],
    ]

    width = 1100
    height = 520
    left = 220
    right = 60
    top = 90
    bottom = 80

    plot_w = width - left - right
    plot_h = height - top - bottom

    # y scale (deltas in percentage points)
    ys = [v for _, ds, df in items for v in (ds, df)]
    y_min = min(ys + [-35.0])
    y_max = max(ys + [15.0])

    def y(v: float) -> float:
        # map y_max..y_min to 0..plot_h
        return (y_max - v) / (y_max - y_min) * plot_h

    def x(i: int, j: int) -> float:
        # i: temp index (0..1), j: metric index (0/1)
        group_w = plot_w / len(items)
        bar_w = group_w * 0.28
        gap = group_w * 0.10
        cx = (i + 0.5) * group_w
        return cx + (j - 0.5) * (bar_w + gap) - bar_w / 2

    style = (
        "<style>"
        ".title{font: 20px sans-serif; font-weight:600;}"
        ".subtitle{font: 12px sans-serif; fill:#333;}"
        ".label{font: 14px sans-serif;}"
        ".tick{font: 12px sans-serif; fill:#333;}"
        ".val{font: 12px monospace; fill:#111;}"
        "</style>"
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        style,
        f'<text class="title" x="{left}" y="34">Decoding sensitivity (Qwen; seeds 1–2): persona effect persists across temperature</text>',
        f'<text class="subtitle" x="{left}" y="58">Bars show persona-mean minus Neutral Re-asking Control at round 5 (ΔSurvival@5) and turn 1 (ΔFail@1).</text>',
    ]

    # axes
    x0 = left
    y0 = top + plot_h
    parts.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{y0}" stroke="#000" stroke-width="1"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{left+plot_w}" y2="{y0}" stroke="#000" stroke-width="1"/>')

    # y=0 line
    y_zero = top + y(0.0)
    parts.append(f'<line x1="{x0}" y1="{y_zero:.2f}" x2="{left+plot_w}" y2="{y_zero:.2f}" stroke="#000" stroke-width="1" opacity="0.35"/>')

    # y ticks
    for t in [-30, -20, -10, 0, 10]:
        yt = top + y(float(t))
        parts.append(f'<line x1="{x0-4}" y1="{yt:.2f}" x2="{x0}" y2="{yt:.2f}" stroke="#000" stroke-width="1"/>')
        parts.append(f'<text class="tick" x="{x0-10}" y="{yt+4:.2f}" text-anchor="end">{t:+d}</text>')

    # legend
    legend_x = left
    legend_y = top - 20
    parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="18" height="12" fill="#1f77b4" opacity="0.90"/>')
    parts.append(f'<text class="tick" x="{legend_x+26}" y="{legend_y+11}">ΔSurvival@5</text>')
    parts.append(f'<rect x="{legend_x+160}" y="{legend_y}" width="18" height="12" fill="#d62728" opacity="0.90"/>')
    parts.append(f'<text class="tick" x="{legend_x+186}" y="{legend_y+11}">ΔFail@1</text>')

    # bars
    colors = ["#1f77b4", "#d62728"]
    names = ["ΔSurvival@5", "ΔFail@1"]
    for i, (temp_label, d_surv, d_fail) in enumerate(items):
        vals = [d_surv, d_fail]
        # x tick label
        group_w = plot_w / len(items)
        cx = left + (i + 0.5) * group_w
        parts.append(f'<text class="label" x="{cx:.2f}" y="{y0+34}" text-anchor="middle">temp={_escape(temp_label)}</text>')

        for j, v in enumerate(vals):
            bx = left + x(i, j)
            by = top + y(max(v, 0.0))
            bh = abs(y(v) - y(0.0))
            parts.append(
                f'<rect x="{bx:.2f}" y="{by:.2f}" width="{(plot_w/len(items))*0.28:.2f}" height="{bh:.2f}" fill="{colors[j]}" opacity="0.90"/>'
            )
            # value label
            vy = (by - 6) if v >= 0 else (by + bh + 16)
            parts.append(
                f'<text class="val" x="{bx + (plot_w/len(items))*0.14:.2f}" y="{vy:.2f}" text-anchor="middle">{v:+.2f}</text>'
            )

    parts.append(f'<text class="subtitle" x="{left}" y="{height-22}">Tracked artifact: docs/paper/artifacts/decoding_sweep_qwen_temp_summary_seed1-2_{DATE_TAG}.csv</text>')

    parts.append("</svg>")

    out = OUT / f"decoding_sweep_qwen_delta_seed1-2_{DATE_TAG}.svg"
    write_svg(out, "\n".join(parts))


if __name__ == "__main__":
    main()
