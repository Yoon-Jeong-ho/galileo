#!/usr/bin/env python3
"""Create ARC (AI2 ARC) MCQA jsonl.

Output format:
{"task":"mcqa","question":"...","choices":[{"label":"A","text":"..."},...],"label":"B"}
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["ARC-Easy", "ARC-Challenge"], default="ARC-Easy")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    if args.out:
        out = args.out
    else:
        suffix = args.subset.lower().replace("-", "").replace(" ", "")
        out = "/data_x/aa007878/galileo/data/arc_" + suffix + ".jsonl"

    ds = load_dataset("ai2_arc", args.subset, split=args.split)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in ds:
            q = (ex.get("question") or "").strip()
            choices = ex.get("choices") or {}
            labels = choices.get("label") or []
            texts = choices.get("text") or []
            answer_key = (ex.get("answerKey") or "").strip().upper()

            if not q or not labels or not texts or not answer_key:
                continue

            ch = []
            for lab, txt in zip(labels, texts):
                lab = str(lab).strip().upper()
                txt = str(txt).strip()
                if lab and txt:
                    ch.append({"label": lab, "text": txt})

            row = {"task": "mcqa", "question": q, "choices": ch, "label": answer_key}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if args.limit > 0 and n >= args.limit:
                break

    print("wrote", out_path, "lines", n)


if __name__ == "__main__":
    main()
