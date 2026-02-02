#!/usr/bin/env python3
"""Create SQuAD-style QA jsonl.

Output format (one task per file):
{"task":"qa","question":"...","answers":[...]}

We include context inside the question to keep the pipeline simple.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data_x/aa007878/galileo/data/squad.jsonl")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    ds = load_dataset("squad", split=args.split)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in ds:
            context = ex.get("context", "").strip()
            q = ex.get("question", "").strip()
            answers = ex.get("answers", {}).get("text", [])
            answers = [a.strip() for a in answers if a and a.strip()]
            if not q or not answers:
                continue

            question = f"Context: {context}\n\nQuestion: {q}"
            row = {"task": "qa", "question": question, "answers": answers}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if args.limit > 0 and n >= args.limit:
                break

    print("wrote", out_path, "lines", n)


if __name__ == "__main__":
    main()
