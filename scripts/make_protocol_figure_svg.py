#!/usr/bin/env python3
"""Generate a simple, submission-ready SVG protocol diagram (stdlib-only).

Output:
- docs/paper/figures/protocol_overview.svg

The goal is a clean 1-column-width diagram showing the 3 phases:
Initial eval -> multi-round persona/control pressure -> recovery.

Usage:
  python3 scripts/make_protocol_figure_svg.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "paper" / "figures" / "protocol_overview.svg"


def esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def box(x: int, y: int, w: int, h: int, title: str, lines: list[str], *, fill: str = "#f7f7f7") -> str:
    pad = 16
    lh = 18
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" ry="14" fill="{fill}" stroke="#111" stroke-width="1.2"/>'
    ]
    parts.append(
        f'<text x="{x+pad}" y="{y+pad+6}" class="title">{esc(title)}</text>'
    )
    ty = y + pad + 28
    for i, ln in enumerate(lines):
        parts.append(
            f'<text x="{x+pad}" y="{ty + i*lh}" class="body">{esc(ln)}</text>'
        )
    return "\n".join(parts)


def arrow(x1: int, y1: int, x2: int, y2: int, label: str | None = None) -> str:
    parts = [f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#111" stroke-width="2" marker-end="url(#arrow)"/>' ]
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2 - 8
        parts.append(f'<text x="{mx}" y="{my}" class="label" text-anchor="middle">{esc(label)}</text>')
    return "\n".join(parts)


def main() -> None:
    w, h = 1100, 520

    style = """
    <style>
      .header{font: 20px sans-serif; font-weight:700; fill:#111;}
      .title{font: 16px sans-serif; font-weight:700; fill:#111;}
      .body{font: 14px sans-serif; fill:#111;}
      .label{font: 13px sans-serif; fill:#111;}
      .small{font: 12px sans-serif; fill:#222;}
    </style>
    """.strip()

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        style,
        """
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L8,3 L0,6 z" fill="#111" />
          </marker>
        </defs>
        """.strip(),
        f'<text x="48" y="44" class="header">GALILEO protocol (ground-truth multi-turn robustness)</text>',
    ]

    # layout
    bx_w, bx_h = 310, 240
    y0 = 110
    xA, xB, xC = 60, 395, 730

    svg.append(
        box(
            xA,
            y0,
            bx_w,
            bx_h,
            "Phase 1: Initial evaluation",
            [
                "Answer once on ground-truth tasks",
                "Score correctness → keep initially-correct set C",
                "Export per-example logs",
            ],
            fill="#eef6ff",
        )
    )

    svg.append(
        box(
            xB,
            y0,
            bx_w,
            bx_h,
            "Phase 2: Multi-round pressure",
            [
                "For each x ∈ C, run up to R rounds",
                "Persona pressure (5 personas) OR",
                "Neutral Re-asking Control (drift baseline)",
                "Measure survival + TOF per round",
            ],
            fill="#fff3e6",
        )
    )

    svg.append(
        box(
            xC,
            y0,
            bx_w,
            bx_h,
            "Phase 3: Recovery",
            [
                "For flipped examples, apply recovery prompt",
                "Score recovery conditional on flip",
                "Report recovery as distinct axis",
            ],
            fill="#eefaf0",
        )
    )

    # arrows
    svg.append(arrow(xA + bx_w, y0 + bx_h / 2, xB, y0 + bx_h / 2, "logs + C"))
    svg.append(arrow(xB + bx_w, y0 + bx_h / 2, xC, y0 + bx_h / 2, "flip set"))

    # bottom legend / outputs
    legend_y = 390
    svg.append(
        '<rect x="60" y="390" width="980" height="96" rx="14" ry="14" fill="#ffffff" stroke="#111" stroke-width="1.0"/>'
    )
    svg.append(
        f'<text x="76" y="418" class="title">Outputs (per run)</text>'
    )
    svg.append(
        f'<text x="76" y="444" class="body">paper_exports/: survival_curve.csv • turn_of_failure.csv • flip_samples.csv • metadata.json • runner_metadata.json</text>'
    )
    svg.append(
        f'<text x="76" y="468" class="body">Tracked artifacts: compact CSV summaries under docs/paper/artifacts/ → submission figures under docs/paper/figures/</text>'
    )

    svg.append("</svg>")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(svg), encoding="utf-8")
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
