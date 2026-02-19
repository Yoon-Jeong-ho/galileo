#!/usr/bin/env python3
"""Paper-oriented exports from Galileo outputs (stdlib only).

Exports:
- survival_curve.csv: persona x round survival rate (aggregate)
- turn_of_failure.csv: distribution of the first incorrect round per persona
- flip_samples.csv: qualitative sample sheet for manual taxonomy labeling

The script reads:
- adversarial_survival.csv (for aggregate curves)
- model_results_dir/*_adversarial.jsonl (for turn-of-failure + flip samples)

Usage:
  python scripts/paper_export.py \
    --results_root /mnt/.../7b \
    --model_dir /mnt/.../7b/Qwen2.5-7B-Instruct \
    --out_dir /mnt/.../7b/paper_exports \
    --num_flip_samples 200 \
    --seed 42
"""

import argparse
import csv
import json
import random
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_persona_id(persona: str) -> str:
    """Normalize persona/control identifiers for paper exports.

    We standardize the neutral drift baseline as:
      persona = "neutral_reask_control"

    This makes plotting/validation stable across old/new runs.
    """
    p = (persona or "").strip()
    low = p.lower()

    # Common historical labels for the neutral re-asking baseline.
    #
    # Note: some runs store the control as a *persona key* (e.g., from CLI flags)
    # rather than a human-readable name. We normalize those too so downstream
    # plotting/aggregation scripts can rely on a single stable identifier.
    if low in {
        "control_reask",
        "control re-asking",
        "control re-ask",
        "neutral re-asking control",
        "neutral reasking control",
        "neutral re-ask control",
        "control",
        "re-asking control",
        "reasking control",
    }:
        return "neutral_reask_control"

    return p


def export_survival_curve(adversarial_survival_csv: Path, out_path: Path):
    adv = read_csv(adversarial_survival_csv)
    # aggregate across datasets: sum survived/total per persona per round
    by_pr = defaultdict(lambda: {"survived": 0, "total": 0})

    for r in adv:
        persona = normalize_persona_id(r["persona"])
        key = (persona, int(r["round"]))
        by_pr[key]["survived"] += int(r["survived"])
        by_pr[key]["total"] += int(r["total"])

    rows = []
    for (persona, rnd), c in sorted(by_pr.items(), key=lambda x: (x[0][0], x[0][1])):
        total = c["total"]
        rate = (c["survived"] / total * 100.0) if total else 0.0
        rows.append(
            {
                "persona": persona,
                "round": rnd,
                "survived": c["survived"],
                "total": total,
                "survival_rate": f"{rate:.6f}",
            }
        )

    write_csv(out_path, ["persona", "round", "survived", "total", "survival_rate"], rows)


def first_failure_turn(turns):
    # turns: list[{turn, is_correct, ...}]
    # returns integer: 1..max_rounds for first incorrect, or 0 if never incorrect
    for t in turns:
        try:
            if not bool(t.get("is_correct")):
                return int(t.get("turn"))
        except Exception:
            continue
    return 0


def export_turn_of_failure(model_dir: Path, out_path: Path):
    # Distribution over first failure turns, per persona and per dataset.
    # Reads *_adversarial.jsonl
    dist = defaultdict(lambda: defaultdict(int))  # (persona,test)-> turn->count

    for jf in sorted(model_dir.glob("*_adversarial.jsonl")):
        test_name = jf.name.replace("_adversarial.jsonl", "")
        for row in iter_jsonl(jf):
            persona = normalize_persona_id(row.get("persona_name") or row.get("persona") or "(unknown)")
            turns = row.get("turns") or []
            ft = first_failure_turn(turns)
            dist[(persona, test_name)][ft] += 1

    # write long-form
    rows = []
    for (persona, test_name), counter in sorted(dist.items(), key=lambda x: (x[0][0], x[0][1])):
        total = sum(counter.values())
        for ft, cnt in sorted(counter.items(), key=lambda x: x[0]):
            label = "never_failed" if ft == 0 else f"fail_at_{ft}"
            rows.append(
                {
                    "persona": persona,
                    "test_name": test_name,
                    "fail_turn": ft,
                    "fail_turn_label": label,
                    "count": cnt,
                    "total": total,
                    "rate": f"{(cnt/total*100.0) if total else 0.0:.6f}",
                }
            )

    write_csv(out_path, ["persona", "test_name", "fail_turn", "fail_turn_label", "count", "total", "rate"], rows)


def export_flip_samples(model_dir: Path, out_path: Path, num_samples: int, seed: int):
    rng = random.Random(seed)

    pool = []
    for jf in sorted(model_dir.glob("*_adversarial.jsonl")):
        test_name = jf.name.replace("_adversarial.jsonl", "")
        for i, row in enumerate(iter_jsonl(jf)):
            turns = row.get("turns") or []
            ft = first_failure_turn(turns)
            if ft == 0:
                continue  # not flipped
            persona = normalize_persona_id(row.get("persona_name") or row.get("persona") or "(unknown)")

            # extract the failure turn info
            fail_turn_obj = None
            for t in turns:
                if int(t.get("turn")) == ft:
                    fail_turn_obj = t
                    break

            pool.append(
                {
                    "test_name": test_name,
                    "persona": persona,
                    "fail_turn": ft,
                    "question": (row.get("question") or "")[:5000],
                    "ground_truth": str(row.get("ground_truth", ""))[:2000],
                    "initial_response": str(row.get("initial_response", ""))[:2000],
                    "fail_adversarial_claim": (fail_turn_obj or {}).get("adversarial_claim", ""),
                    "fail_model_response": (fail_turn_obj or {}).get("model_response", ""),
                    "fail_extracted_answer": (fail_turn_obj or {}).get("extracted_answer", ""),
                    # Manual taxonomy label fields (fill later)
                    "taxonomy_label": "",
                    "notes": "",
                }
            )

    rng.shuffle(pool)
    pool = pool[: max(0, num_samples)]

    write_csv(
        out_path,
        [
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
        ],
        pool,
    )


def _safe_git_commit(cwd: Path):
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd), stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def write_metadata(out_dir: Path, args, git_commit: str | None):
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool": "scripts/paper_export.py",
        "git_commit": git_commit,
        "results_root": str(Path(args.results_root).resolve()),
        "model_dir": str(Path(args.model_dir).resolve()),
        "seed": int(args.seed),
        "num_flip_samples": int(args.num_flip_samples),
        # NOTE: decoding params are owned by the runner; this export script only records what it can see.
        "notes": "Decoding params should be recorded by the experiment runner (e.g., results/<run>/paper_exports/metadata.json).",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True, help="Path containing *accuracy.csv files")
    ap.add_argument("--model_dir", required=True, help="Path containing *_adversarial.jsonl logs")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_flip_samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    results_root = Path(args.results_root)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    git_commit = _safe_git_commit(Path(__file__).resolve().parent.parent)

    export_survival_curve(results_root / "adversarial_survival.csv", out_dir / "survival_curve.csv")
    export_turn_of_failure(model_dir, out_dir / "turn_of_failure.csv")
    export_flip_samples(model_dir, out_dir / "flip_samples.csv", num_samples=args.num_flip_samples, seed=args.seed)
    write_metadata(out_dir, args, git_commit)

    print("Wrote:")
    print(out_dir / "survival_curve.csv")
    print(out_dir / "turn_of_failure.csv")
    print(out_dir / "flip_samples.csv")
    print(out_dir / "metadata.json")


if __name__ == "__main__":
    main()
