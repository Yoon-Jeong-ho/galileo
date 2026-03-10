#!/usr/bin/env python3
"""Render a simple SVG trade-off plot from comparison CSV rows.

Expected columns:
- dataset
- delta_authority_survival_r5
- delta_authority_post_recovery_acc
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


COLORS = {
    "gsm8k": "#2563eb",
    "arc_easy_val_50": "#dc2626",
}

SHAPES = {
    "grounded": "circle",
    "evidence_gate": "square",
}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grounded_csv", required=True)
    ap.add_argument("--gate_csv", required=True)
    ap.add_argument("--out_svg", required=True)
    args = ap.parse_args()

    rows = []
    for kind, csv_path in [("grounded", Path(args.grounded_csv)), ("evidence_gate", Path(args.gate_csv))]:
        for row in read_csv(csv_path):
            row = dict(row)
            row["kind"] = kind
            rows.append(row)

    width = 900
    height = 520
    left = 110
    right = 70
    top = 70
    bottom = 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    xs = [float(r["delta_authority_survival_r5"]) for r in rows]
    ys = [float(r["delta_authority_post_recovery_acc"]) for r in rows]
    x_min = min(xs + [-0.05]) - 0.02
    x_max = max(xs + [0.15]) + 0.02
    y_min = min(ys + [-0.05]) - 0.02
    y_max = max(ys + [0.05]) + 0.02

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - (y - y_min) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif}.axis{stroke:#444;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.title{font-size:20px;font-weight:bold}.label{font-size:14px}.small{font-size:12px}</style>",
        '<text x="30" y="34" class="title">Mitigation / correction trade-off (vs evidence baseline)</text>',
    ]

    # axes
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>')
    x_zero = sx(0.0)
    y_zero = sy(0.0)
    parts.append(f'<line x1="{x_zero}" y1="{top}" x2="{x_zero}" y2="{top+plot_h}" class="grid"/>')
    parts.append(f'<line x1="{left}" y1="{y_zero}" x2="{left+plot_w}" y2="{y_zero}" class="grid"/>')

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x_val = x_min + frac * (x_max - x_min)
        x = sx(x_val)
        parts.append(f'<line x1="{x}" y1="{top+plot_h}" x2="{x}" y2="{top+plot_h+6}" class="axis"/>')
        parts.append(f'<text x="{x}" y="{top+plot_h+24}" text-anchor="middle" class="small">{x_val:.2f}</text>')
        y_val = y_min + frac * (y_max - y_min)
        y = sy(y_val)
        parts.append(f'<line x1="{left-6}" y1="{y}" x2="{left}" y2="{y}" class="axis"/>')
        parts.append(f'<text x="{left-10}" y="{y+4}" text-anchor="end" class="small">{y_val:.2f}</text>')

    parts.append(f'<text x="{left+plot_w/2}" y="{height-22}" text-anchor="middle" class="label">Authority survival@5 improvement vs evidence baseline</text>')
    parts.append(f'<text x="24" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 24 {top+plot_h/2})" class="label">Authority PostRecoveryAcc change vs evidence baseline</text>')

    for row in rows:
        dataset = row["dataset"]
        kind = row["kind"]
        x = sx(float(row["delta_authority_survival_r5"]))
        y = sy(float(row["delta_authority_post_recovery_acc"]))
        color = COLORS.get(dataset, "#555")
        if SHAPES[kind] == "circle":
            parts.append(f'<circle cx="{x}" cy="{y}" r="8" fill="{color}" opacity="0.85"/>')
        else:
            parts.append(f'<rect x="{x-8}" y="{y-8}" width="16" height="16" fill="{color}" opacity="0.85"/>')
        label = ("ARC" if dataset == "arc_easy_val_50" else "GSM8K") + " / " + kind
        parts.append(f'<text x="{x+12}" y="{y-10}" class="small">{label}</text>')

    legend_y = 56
    parts.append(f'<circle cx="{left}" cy="{legend_y}" r="7" fill="#666"/><text x="{left+14}" y="{legend_y+4}" class="small">grounded</text>')
    parts.append(f'<rect x="{left+120-7}" y="{legend_y-7}" width="14" height="14" fill="#666"/><text x="{left+134}" y="{legend_y+4}" class="small">evidence_gate</text>')
    parts.append(f'<text x="{width-250}" y="{legend_y+4}" class="small">Blue: GSM8K | Red: ARC-Easy</text>')
    parts.append("</svg>")

    Path(args.out_svg).write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote: {args.out_svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
