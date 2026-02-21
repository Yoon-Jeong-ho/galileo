#!/usr/bin/env python3
r"""Generate LaTeX rows for the main results table (Table 1) from tracked CSV artifacts.

Scope (current):
- Uses per-model survival summaries under docs/paper/artifacts/tier1_*_survival_summary_*.csv
- Computes persona-weighted aggregates as an equal-weight mean across personas (excluding NRC) of
  persona--NRC deltas, then reconstructs the Persona absolute as NRC + mean(delta).
  - Survival@5: outputs NRC mean±std, Persona mean±std (std over personas of the delta at this
    aggregation level), and Δ = Persona − NRC.

Limitations:
- Many summaries currently do not include absolute Fail@1 values (only persona--NRC deltas).
  The paper table therefore reports $\Delta$Fail@1 only.

Usage:
  python3 scripts/gen_latex_table1_from_artifacts.py \
    --out docs/paper/latex_paper_emnlp2023/generated/table1_rows.tex
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "docs" / "paper" / "artifacts"


@dataclass(frozen=True)
class Row:
    model: str
    # Survival@5
    nrc_surv_mean: float
    nrc_surv_std: float
    persona_surv_mean: float
    persona_surv_std: float
    # Fail@1 (often only delta available)
    delta_fail1_mean: float | None = None
    delta_fail1_std: float | None = None
    # Recovery@flip (collapsed)
    nrc_rec_mean: float | None = None
    nrc_rec_std: float | None = None
    persona_rec_mean: float | None = None
    persona_rec_std: float | None = None
    delta_rec_mean: float | None = None


def _read_survival_summary(path: Path) -> tuple[str, list[dict[str, str]]]:
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    model = rows[0]["model"].strip()
    return model, rows


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def _std(xs: Iterable[float]) -> float:
    xs = list(xs)
    if len(xs) <= 1:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def _fmt(x: float, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "--"
    return f"{x:.{digits}f}"


def _find_latest(pattern: str) -> Path:
    candidates = sorted(ARTIFACTS_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No artifacts match {pattern} under {ARTIFACTS_DIR}")
    # Prefer lexicographically last (date suffix in filename).
    return candidates[-1]


def build_rows(model_to_glob: dict[str, str]) -> list[Row]:
    out: list[Row] = []
    for display, glob_pat in model_to_glob.items():
        p = _find_latest(glob_pat)
        _model, rows = _read_survival_summary(p)

        # NRC row
        nrc = [r for r in rows if r["persona"].strip() == "neutral_reask_control"]
        if len(nrc) != 1:
            raise ValueError(f"Expected exactly 1 NRC row in {p}, got {len(nrc)}")
        nrc_surv_mean = float(nrc[0]["survival_r5_mean"])
        nrc_surv_std = float(nrc[0].get("survival_r5_std", 0.0) or 0.0)

        persona_rows = [r for r in rows if r["persona"].strip() != "neutral_reask_control"]

        # Align with the paper's "persona-weighted aggregate" definition: aggregate the
        # persona--NRC deltas (equal-weight over personas), then reconstruct the Persona
        # absolute value as NRC + mean(delta).
        surv_deltas = [float(r["survival_r5_mean"]) - nrc_surv_mean for r in persona_rows]
        delta_surv_mean = _mean(surv_deltas)
        delta_surv_std = _std(surv_deltas)
        persona_surv_mean = nrc_surv_mean + delta_surv_mean
        # Since Persona = NRC + Δ, the dispersion we can report at this aggregation level
        # is the dispersion of Δ across personas.
        persona_surv_std = delta_surv_std

        # Fail@1: many summaries only provide delta_fail_r1_mean/std.
        fail_deltas = [
            float(r["delta_fail_r1_mean"])
            for r in persona_rows
            if r.get("delta_fail_r1_mean") not in (None, "")
        ]
        fail_delta_mean = _mean(fail_deltas) if fail_deltas else None
        fail_delta_std = _std(fail_deltas) if len(fail_deltas) >= 2 else (0.0 if fail_deltas else None)

        out.append(
            Row(
                model=display,
                nrc_surv_mean=nrc_surv_mean,
                nrc_surv_std=nrc_surv_std,
                persona_surv_mean=persona_surv_mean,
                persona_surv_std=persona_surv_std,
                delta_fail1_mean=fail_delta_mean,
                delta_fail1_std=fail_delta_std,
            )
        )
    return out


def render_table_rows(rows: list[Row]) -> str:
    lines: list[str] = []
    # Keep the output strictly to tabular rows (no comments/blank lines),
    # because this file is \input'ed inside a tabularx environment.
    for r in rows:
        delta_surv = r.persona_surv_mean - r.nrc_surv_mean

        # Fail@1: we typically only have deltas from tier1 summaries.
        if r.delta_fail1_mean is None:
            fail_delta_cell = "--"
        else:
            if r.delta_fail1_std is None:
                fail_delta_cell = _fmt(r.delta_fail1_mean)
            else:
                fail_delta_cell = f"{_fmt(r.delta_fail1_mean)}$\\pm${_fmt(r.delta_fail1_std)}"

        # Recovery@flip (collapsed): only tracked for some settings.
        rec_nrc_cell = "--" if r.nrc_rec_mean is None else f"{_fmt(r.nrc_rec_mean)}$\\pm${_fmt(r.nrc_rec_std or 0.0)}"
        rec_p_cell = "--" if r.persona_rec_mean is None else f"{_fmt(r.persona_rec_mean)}$\\pm${_fmt(r.persona_rec_std or 0.0)}"
        rec_d_cell = "--" if r.delta_rec_mean is None else _fmt(r.delta_rec_mean)

        lines.append(
            " ".join(
                [
                    f"{r.model}",
                    "&",
                    f"{_fmt(r.nrc_surv_mean)}$\\pm${_fmt(r.nrc_surv_std)}",
                    "&",
                    f"{_fmt(r.persona_surv_mean)}$\\pm${_fmt(r.persona_surv_std)}",
                    "&",
                    f"{_fmt(delta_surv)}",
                    "&",
                    f"{fail_delta_cell}",
                    "&",
                    f"{rec_nrc_cell} & {rec_p_cell} & {rec_d_cell}",
                    "\\\\",
                ]
            )
        )

    # Put \bottomrule inside the input to avoid alignment edge cases.
    lines.append("\\bottomrule")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        required=True,
        help="Output .tex file path (relative to repo root or absolute)",
    )
    args = ap.parse_args()

    # Keep this list aligned with Table~\ref{tab:main} in main.tex.
    # Qwen2.5-7B: we currently track a paper-facing persona-weighted aggregate in Table W.
    # Other families: use the tier1_*_survival_summary_*.csv artifacts.
    model_to_glob = {
        "Qwen2.5-14B-Instruct": "tier1_qwen2p5_14b_seed1-2_survival_summary_*.csv",
        "Llama-3.1-8B-Instruct": "tier1_llama3_8b_seed1-2_survival_summary_*.csv",
        "Llama-3.2-3B-Instruct": "tier1_llama3_3b_seed1-2_survival_summary_*.csv",
        "Mistral-7B-Instruct": "tier1_mistral7b_seed1-2_survival_summary_*.csv",
        "Mistral-Nemo-Instruct": "tier1_mistralnemo_seed1-2_survival_summary_*.csv",
        "Phi-3-mini-4k-instruct": "tier1_phi3mini_seed1-2_survival_summary_*.csv",
        "Phi-3.5-mini-instruct": "tier1_phi35mini_seed1-2_survival_summary_*.csv",
        "Zephyr-7B": "tier1_zephyr7b_seed1-2_survival_summary_*.csv",
        "DeepSeek-LLM-7B-Chat": "tier1_deepseek7b_seed1-2_survival_summary_*.csv",
        "Yi-6B-Chat": "tier1_yi6b_seed1-2_survival_summary_*.csv",
    }

    rows = []

    # Special-case: Qwen2.5-7B from Table W persona-weighted aggregate.
    table_w = _find_latest("table_w_control_vs_persona_seed1-4_mean_std_*.csv")
    with table_w.open("r", newline="") as f:
        tw_rows = list(csv.DictReader(f))
    tw_surv = [r for r in tw_rows if r["metric"].strip() == "Survival@5"]
    if len(tw_surv) != 1:
        raise ValueError(f"Expected exactly 1 Survival@5 row in {table_w}")
    # Also pull Fail@1 from Table W (percentage units).
    tw_fail1 = [r for r in tw_rows if r["metric"].strip() == "Fail@1"]
    if len(tw_fail1) != 1:
        raise ValueError(f"Expected exactly 1 Fail@1 row in {table_w}")

    # Recovery@flip (collapsed) is tracked in a separate artifact CSV (percent units).
    rec_csv = _find_latest("recovery_collapsed_control_vs_persona_seed1-4_mean_std_*.csv")
    with rec_csv.open("r", newline="") as f:
        rec_rows = list(csv.DictReader(f))
    if not rec_rows:
        raise ValueError(f"Empty CSV: {rec_csv}")
    rec = rec_rows[0]

    rows.append(
        Row(
            model="Qwen2.5-7B-Instruct",
            nrc_surv_mean=float(tw_surv[0]["control_mean"]) / 100.0,
            nrc_surv_std=float(tw_surv[0]["control_std"]) / 100.0,
            persona_surv_mean=float(tw_surv[0]["persona_weighted_mean"]) / 100.0,
            persona_surv_std=float(tw_surv[0]["persona_weighted_std"]) / 100.0,
            delta_fail1_mean=(float(tw_fail1[0]["persona_weighted_mean"]) - float(tw_fail1[0]["control_mean"])) / 100.0,
            delta_fail1_std=float(tw_fail1[0]["persona_weighted_std"]) / 100.0,
            nrc_rec_mean=float(rec["control_mean"]) / 100.0,
            nrc_rec_std=float(rec["control_std"]) / 100.0,
            persona_rec_mean=float(rec["persona_pressure_mean"]) / 100.0,
            persona_rec_std=float(rec["persona_pressure_std"]) / 100.0,
            delta_rec_mean=float(rec["delta_persona_minus_control_mean"]) / 100.0,
        )
    )

    rows.extend(build_rows(model_to_glob))
    tex = render_table_rows(rows)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
