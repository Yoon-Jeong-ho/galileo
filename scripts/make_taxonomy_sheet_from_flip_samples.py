#!/usr/bin/env python3
"""Create a balanced qualitative taxonomy labeling sheet from paper_exports/flip_samples.csv.

Why:
- Some paper SSOT run roots under results_paper/ may not include the full *_adversarial.jsonl logs,
  but they *do* include paper-ready flip_samples.csv.
- This script samples flipped cases for manual taxonomy labeling in a reviewer-auditable way.

Input:
- One or more CSVs produced by scripts/paper_export.py (paper_exports/flip_samples.csv).

Output:
- A CSV suitable for manual labeling, with empty columns taxonomy_label, notes.

Sampling strategy:
- Balanced across (task_group inferred from test_name) and persona.
- per_cell = examples per (task_group, persona).

Example:
  python3 scripts/make_taxonomy_sheet_from_flip_samples.py \
    --flip_csvs _tmp/flip_s1.csv,_tmp/flip_s2.csv,_tmp/flip_s3.csv,_tmp/flip_s4.csv \
    --out_csv docs/paper/artifacts/taxonomy_labeling_sheet_qwen_seed1-4_20260217.csv \
    --per_cell 10
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def task_group(test_name: str) -> str:
    t = (test_name or "").lower()
    if "gsm8k" in t or "svamp" in t:
        return "math"
    if "arc" in t:
        return "mcqa"
    if "triviaqa" in t:
        return "openqa"
    if "squad" in t:
        return "qa"
    return "other"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flip_csvs", required=True, help="Comma-separated flip_samples.csv paths")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--per_cell", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    flip_paths = [Path(p.strip()) for p in args.flip_csvs.split(",") if p.strip()]
    if not flip_paths:
        raise SystemExit("no flip_csvs provided")

    pool: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)  # (task_group, persona) -> rows

    for p in flip_paths:
        if not p.exists():
            raise SystemExit(f"missing flip csv: {p}")
        for r in read_csv(p):
            tn = (r.get("test_name") or "").strip()
            persona = (r.get("persona") or "").strip() or "(unknown)"
            tg = task_group(tn)

            # Normalize fields + ensure taxonomy cols exist.
            pool[(tg, persona)].append(
                {
                    "task_group": tg,
                    "test_name": tn,
                    "persona": persona,
                    "fail_turn": (r.get("fail_turn") or "").strip(),
                    "question": (r.get("question") or "")[:5000],
                    "ground_truth": (r.get("ground_truth") or "")[:2000],
                    "initial_response": (r.get("initial_response") or "")[:2000],
                    "fail_adversarial_claim": (r.get("fail_adversarial_claim") or "")[:2000],
                    "fail_model_response": (r.get("fail_model_response") or "")[:2000],
                    "fail_extracted_answer": (r.get("fail_extracted_answer") or "")[:2000],
                    "taxonomy_label": "",
                    "notes": "",
                }
            )

    rows: list[dict[str, str]] = []
    task_groups = sorted({k[0] for k in pool.keys()})
    personas = sorted({k[1] for k in pool.keys()})

    for tg in task_groups:
        for persona in personas:
            cell = pool.get((tg, persona), [])
            if not cell:
                continue
            rng.shuffle(cell)
            rows.extend(cell[: args.per_cell])

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "task_group",
        "test_name",
        "persona",
        "fail_turn",
        "question",
        "ground_truth",
        "initial_response",
        "fail_adversarial_claim",
        "fail_model_response",
        "fail_extracted_answer",
        "taxonomy_label",
        "notes",
    ]

    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[OK] wrote {out} rows={len(rows)} per_cell={args.per_cell} cells={len(pool)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
