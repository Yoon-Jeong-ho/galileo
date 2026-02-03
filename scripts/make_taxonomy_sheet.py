#!/usr/bin/env python3
"""Create a balanced qualitative taxonomy labeling sheet.

Reads *_adversarial.jsonl logs and samples flipped cases.

Output CSV includes empty columns: taxonomy_label, notes.

Sampling strategy:
- balanced across (task_type inferred from test_name) and persona
- per_cell = examples per (task_group, persona)
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def first_failure_turn(turns):
    for t in turns or []:
        try:
            if not bool(t.get("is_correct")):
                return int(t.get("turn"))
        except Exception:
            continue
    return 0


def task_group(test_name: str) -> str:
    t = test_name.lower()
    if "gsm8k" in t or "svamp" in t:
        return "math"
    if "arc" in t:
        return "mcqa"
    if "triviaqa" in t:
        return "openqa"
    if "squad" in t:
        return "qa"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="dir containing *_adversarial.jsonl")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--per_cell", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    model_dir = Path(args.model_dir)

    pool = defaultdict(list)  # (task_group, persona) -> list[row]

    for jf in sorted(model_dir.glob("*_adversarial.jsonl")):
        test_name = jf.name.replace("_adversarial.jsonl", "")
        tg = task_group(test_name)
        for row in iter_jsonl(jf):
            turns = row.get("turns") or []
            ft = first_failure_turn(turns)
            if ft == 0:
                continue
            persona = row.get("persona_name") or row.get("persona") or "(unknown)"

            fail_turn_obj = None
            for t in turns:
                if int(t.get("turn")) == ft:
                    fail_turn_obj = t
                    break

            pool[(tg, persona)].append({
                "task_group": tg,
                "test_name": test_name,
                "persona": persona,
                "fail_turn": ft,
                "question": (row.get("question") or "")[:5000],
                "ground_truth": str(row.get("ground_truth", ""))[:2000],
                "initial_response": str(row.get("initial_response", ""))[:2000],
                "fail_adversarial_claim": (fail_turn_obj or {}).get("adversarial_claim", ""),
                "fail_model_response": (fail_turn_obj or {}).get("model_response", ""),
                "fail_extracted_answer": (fail_turn_obj or {}).get("extracted_answer", ""),
                "taxonomy_label": "",
                "notes": "",
            })

    rows = []
    task_groups = sorted({k[0] for k in pool.keys()})
    personas = sorted({k[1] for k in pool.keys()})

    for tg in task_groups:
        for persona in personas:
            cell = pool.get((tg, persona), [])
            if not cell:
                continue
            rng.shuffle(cell)
            rows.extend(cell[: args.per_cell])

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "task_group","test_name","persona","fail_turn",
        "question","ground_truth","initial_response",
        "fail_adversarial_claim","fail_model_response","fail_extracted_answer",
        "taxonomy_label","notes",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("wrote", out_path, "rows", len(rows))


if __name__ == "__main__":
    main()
