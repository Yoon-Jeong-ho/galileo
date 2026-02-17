#!/usr/bin/env python3
"""Summarize manually-labeled flip taxonomy sheets into a small CSV.

Input:
- A labeled taxonomy sheet CSV (expects columns incl. task_group, persona, taxonomy_label).

Output:
- A compact CSV of counts and percentages by persona × task_group × taxonomy_label.

This is diagnostic-only; it does not modify any primary evaluator-based metrics.

Example:
  python3 scripts/summarize_taxonomy_labels.py \
    --in_csv docs/paper/artifacts/taxonomy_labeling_sheet_from_flip_samples_qwen_persona_seed1-4_20260217.csv \
    --schema_md docs/paper/artifacts/taxonomy_label_schema_v1_20260218.md \
    --out_csv docs/paper/artifacts/taxonomy_label_breakdown_qwen_persona_seed1-4_YYYYMMDD.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_schema_labels(schema_md: Path) -> list[str]:
    # Extract labels listed as `1) `label`` etc.
    txt = schema_md.read_text(encoding="utf-8", errors="ignore")
    labels = re.findall(r"^\s*\d+\)\s+`([^`]+)`\s*$", txt, flags=re.MULTILINE)
    # Fallback: any backticked tokens under "Labels" section.
    if not labels:
        labels = re.findall(r"`([a-z0-9_]+)`", txt)
    # De-dup preserving order.
    seen = set()
    out = []
    for l in labels:
        if l not in seen:
            seen.add(l)
            out.append(l)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--schema_md", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--include_unlabeled", action="store_true", help="Include empty taxonomy_label rows")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    schema_md = Path(args.schema_md)
    out_csv = Path(args.out_csv)

    rows = read_csv(in_csv)
    allowed = parse_schema_labels(schema_md)
    allowed_set = set(allowed)

    # Validate + bucket.
    bucket: dict[tuple[str, str, str], int] = defaultdict(int)  # (persona, task_group, label)
    totals: dict[tuple[str, str], int] = defaultdict(int)  # (persona, task_group) -> n

    bad = []
    for r in rows:
        persona = (r.get("persona") or "").strip() or "(unknown)"
        tg = (r.get("task_group") or "").strip() or "(unknown)"
        label = (r.get("taxonomy_label") or "").strip()
        if not label:
            if not args.include_unlabeled:
                continue
            label = "(unlabeled)"
        elif label not in allowed_set:
            bad.append(label)
        bucket[(persona, tg, label)] += 1
        totals[(persona, tg)] += 1

    if bad:
        bad_counts = Counter(bad).most_common(10)
        raise SystemExit(f"found taxonomy_label values not in schema: {bad_counts}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "persona",
        "task_group",
        "taxonomy_label",
        "count",
        "pct_within_persona_task_group",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for (persona, tg, label), c in sorted(bucket.items()):
            denom = totals[(persona, tg)]
            pct = (100.0 * c / denom) if denom else 0.0
            w.writerow(
                {
                    "persona": persona,
                    "task_group": tg,
                    "taxonomy_label": label,
                    "count": str(c),
                    "pct_within_persona_task_group": f"{pct:.2f}",
                }
            )

    print(f"[OK] wrote {out_csv} rows={len(bucket)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
