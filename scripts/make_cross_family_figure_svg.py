#!/usr/bin/env python3
"""Make a compact cross-family generalization figure (submission-ready SVG).

We intentionally avoid matplotlib to keep the writing environment lightweight.

Input (tracked artifacts):
- docs/paper/artifacts/tier1_mistral7b_seed1-2_survival_summary_20260210.csv
- docs/paper/artifacts/tier1_llama3_8b_seed1-2_survival_summary_20260210.csv
- docs/paper/artifacts/tier1_llama3_3b_seed1-2_survival_summary_20260212.csv
- docs/paper/artifacts/tier1_phi3mini_seed1-2_survival_summary_20260217.csv
- docs/paper/artifacts/tier1_mistralnemo_seed1-2_survival_summary_20260217.csv
- docs/paper/artifacts/tier1_zephyr7b_seed1-2_survival_summary_20260218.csv
- docs/paper/artifacts/tier1_qwen2p5_14b_seed1-2_survival_summary_20260219.csv
- docs/paper/artifacts/tier1_deepseek7b_seed1-2_survival_summary_20260221.csv

Output:
- docs/paper/figures/cross_family_survival_r5_control_vs_logicaltrap_seed1-2_20260221.svg

Plot:
- For each model family: Survival@5 for Neutral Re-asking Control vs Logical Trap persona.

Usage:
  python3 scripts/make_cross_family_figure_svg.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "docs" / "paper" / "artifacts"
OUT = ROOT / "docs" / "paper" / "figures"
DATE_TAG = "20260221"


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


def _pick(rows: List[Dict[str, str]], persona: str) -> Tuple[float, float]:
    # returns (mean, std) for survival@5
    for r in rows:
        if r["persona"] == persona:
            return float(r["survival_r5_mean"]), float(r["survival_r5_std"])
    raise KeyError(f"persona not found: {persona}")


def write_svg(path: Path, svg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def make_svg(items: List[Tuple[str, float, float, float, float]]) -> str:
    """items: [(family_label, control_mean, control_std, trap_mean, trap_std), ...]"""

    width = 1100
    row_h = 64
    left_margin = 280
    right_margin = 60
    top_margin = 90
    bottom_margin = 70

    n = len(items)
    height = top_margin + bottom_margin + n * row_h

    # scale bars by max mean
    max_v = max([max(c, t) for _, c, _, t, _ in items] + [1e-9])
    plot_w = width - left_margin - right_margin

    def w(v: float) -> float:
        return (v / max_v) * plot_w

    style = (
        "<style>"
        ".title{font: 20px sans-serif; font-weight:600;}"
        ".label{font: 14px sans-serif;}"
        ".small{font: 12px sans-serif; fill:#333;}"
        ".val{font: 12px monospace; fill:#111;}"
        "</style>"
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        style,
        f'<text class="title" x="{left_margin}" y="34">Cross-family Survival@5 (control vs Logical Trap) — seeds 1–2</text>',
        f'<text class="small" x="{left_margin}" y="58">Neutral Re-asking Control is the drift baseline; Logical Trap is a strong pressure persona.</text>',
    ]

    # legend
    legend_y = top_margin - 26
    parts.append(f'<rect x="{left_margin}" y="{legend_y}" width="18" height="12" fill="#9e9e9e" opacity="0.95"/>')
    parts.append(f'<text class="small" x="{left_margin+26}" y="{legend_y+11}">control</text>')
    parts.append(f'<rect x="{left_margin+110}" y="{legend_y}" width="18" height="12" fill="#d62728" opacity="0.90"/>')
    parts.append(f'<text class="small" x="{left_margin+136}" y="{legend_y+11}">Logical Trap</text>')

    # x-axis ticks (0, 50%, 100% of max)
    axis_y = height - bottom_margin + 6
    parts.append(
        f'<line x1="{left_margin}" y1="{axis_y}" x2="{left_margin+plot_w}" y2="{axis_y}" stroke="#000" stroke-width="1"/>'
    )
    for frac in [0.0, 0.5, 1.0]:
        xt = left_margin + frac * plot_w
        val = frac * max_v
        parts.append(f'<line x1="{xt:.2f}" y1="{axis_y}" x2="{xt:.2f}" y2="{axis_y+6}" stroke="#000" stroke-width="1"/>')
        parts.append(f'<text class="small" x="{xt:.2f}" y="{axis_y+22}" text-anchor="middle">{val*100:.0f}%</text>')

    # rows
    for i, (fam, c_mean, c_std, t_mean, t_std) in enumerate(items):
        y0 = top_margin + i * row_h
        y_label = y0 + 22
        parts.append(
            f'<text class="label" x="{left_margin-10}" y="{y_label}" text-anchor="end">{_escape(fam)}</text>'
        )

        # bar positions within row
        bar_h = 14
        y_control = y0 + 8
        y_trap = y0 + 30

        # bars
        parts.append(
            f'<rect x="{left_margin}" y="{y_control}" width="{w(c_mean):.2f}" height="{bar_h}" fill="#9e9e9e" opacity="0.95"/>'
        )
        parts.append(
            f'<rect x="{left_margin}" y="{y_trap}" width="{w(t_mean):.2f}" height="{bar_h}" fill="#d62728" opacity="0.90"/>'
        )

        # values (mean±std)
        parts.append(
            f'<text class="val" x="{left_margin + w(c_mean) + 8:.2f}" y="{y_control+11}" text-anchor="start">{c_mean*100:.2f}±{c_std*100:.2f}</text>'
        )
        parts.append(
            f'<text class="val" x="{left_margin + w(t_mean) + 8:.2f}" y="{y_trap+11}" text-anchor="start">{t_mean*100:.2f}±{t_std*100:.2f}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    # Note: different families were generated on different dates; the figure is regenerated from the
    # tracked summary CSVs listed above.
    mistral = read_csv(ART / "tier1_mistral7b_seed1-2_survival_summary_20260210.csv")
    llama8b = read_csv(ART / "tier1_llama3_8b_seed1-2_survival_summary_20260210.csv")
    llama3b = read_csv(ART / "tier1_llama3_3b_seed1-2_survival_summary_20260212.csv")
    phi3mini = read_csv(ART / "tier1_phi3mini_seed1-2_survival_summary_20260217.csv")
    phi35mini = read_csv(ART / "tier1_phi35mini_seed1-2_survival_summary_20260219.csv")
    mistralnemo = read_csv(ART / "tier1_mistralnemo_seed1-2_survival_summary_20260217.csv")
    zephyr7b = read_csv(ART / "tier1_zephyr7b_seed1-2_survival_summary_20260218.csv")
    qwen14b = read_csv(ART / "tier1_qwen2p5_14b_seed1-2_survival_summary_20260219.csv")
    deepseek7b = read_csv(ART / "tier1_deepseek7b_seed1-2_survival_summary_20260221.csv")

    items = []
    for name, rows in [
        ("Mistral-7B-Instruct v0.3", mistral),
        ("Mistral-Nemo-Instruct-2407", mistralnemo),
        ("Llama-3.1-8B-Instruct", llama8b),
        ("Llama-3.2-3B-Instruct", llama3b),
        ("Phi-3-mini-Instruct", phi3mini),
        ("Phi-3.5-mini-Instruct", phi35mini),
        ("Zephyr-7B-beta", zephyr7b),
        ("DeepSeek-LLM-7B-Chat", deepseek7b),
        ("Qwen2.5-14B-Instruct", qwen14b),
    ]:
        c_mean, c_std = _pick(rows, "neutral_reask_control")
        t_mean, t_std = _pick(rows, "Logical Trap")
        items.append((name, c_mean, c_std, t_mean, t_std))

    svg = make_svg(items)
    out = OUT / f"cross_family_survival_r5_control_vs_logicaltrap_seed1-2_{DATE_TAG}.svg"
    write_svg(out, svg)


if __name__ == "__main__":
    main()
