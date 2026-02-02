#!/usr/bin/env python3
"""Build full QA/MCQA datasets for Galileo.

Writes full (or user-limited) jsonl datasets into an output directory.
We keep task fields so run_experiment.py can run all files together.

Examples:
  python scripts/make_qa_full.py --out_dir /data_x/aa007878/galileo/data_qa_full --seed 42

SQuAD:
  - v1.1: squad
  - v2.0: squad_v2

TriviaQA:
  - default subset=rc (lighter)
"""

import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_dataset


def write_jsonl(rows, out_path: Path, limit: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if limit > 0 and n >= limit:
                break
    return n


def make_squad(out_path: Path, version: str, split: str, limit: int):
    name = "squad" if version == "1.1" else "squad_v2"
    ds = load_dataset(name, split=split)

    rows = []
    for ex in ds:
        context = (ex.get("context") or "").strip()
        q = (ex.get("question") or "").strip()
        answers = (ex.get("answers") or {}).get("text", [])
        answers = [a.strip() for a in answers if a and str(a).strip()]
        if not q or not answers:
            continue
        question = f"Context: {context}\n\nQuestion: {q}" if context else q
        rows.append({"task": "qa", "question": question, "answers": answers})

    return write_jsonl(rows, out_path, limit)


def make_arc(out_path: Path, subset: str, split: str, limit: int):
    ds = load_dataset("ai2_arc", subset, split=split)
    rows = []
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
        if not ch:
            continue
        rows.append({"task": "mcqa", "question": q, "choices": ch, "label": answer_key})

    return write_jsonl(rows, out_path, limit)


def make_triviaqa(out_path: Path, subset: str, split: str, limit: int):
    ds = load_dataset("trivia_qa", subset, split=split)
    rows = []
    for ex in ds:
        q = (ex.get("question") or "").strip()
        ans = ex.get("answer") or {}
        aliases = []
        if isinstance(ans, dict):
            if ans.get("value"):
                aliases.append(str(ans["value"]))
            aliases.extend([str(a) for a in (ans.get("aliases") or [])])
        aliases = [a.strip() for a in aliases if a and a.strip()]
        seen = set()
        uniq = []
        for a in aliases:
            k = a.lower()
            if k in seen:
                continue
            seen.add(k)
            uniq.append(a)
        if not q or not uniq:
            continue
        rows.append({"task": "qa", "question": q, "answers": uniq})

    return write_jsonl(rows, out_path, limit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=-1, help="If set, cap each dataset to N examples")

    ap.add_argument("--squad_splits", default="validation", help="comma-separated")
    ap.add_argument("--triviaqa_subset", default="rc")
    ap.add_argument("--triviaqa_split", default="validation")

    ap.add_argument("--arc_subset", default="ARC-Easy")
    ap.add_argument("--arc_split", default="validation")

    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # SQuAD v1.1 + v2.0
    for split in [s.strip() for s in args.squad_splits.split(",") if s.strip()]:
        n = make_squad(out_dir / f"squad11_{split}.jsonl", version="1.1", split=split, limit=args.limit)
        print("wrote", out_dir / f"squad11_{split}.jsonl", n)
        n = make_squad(out_dir / f"squad20_{split}.jsonl", version="2.0", split=split, limit=args.limit)
        print("wrote", out_dir / f"squad20_{split}.jsonl", n)

    # ARC
    n = make_arc(out_dir / f"arc_easy_{args.arc_split}.jsonl", subset=args.arc_subset, split=args.arc_split, limit=args.limit)
    print("wrote", out_dir / f"arc_easy_{args.arc_split}.jsonl", n)

    # TriviaQA
    n = make_triviaqa(out_dir / f"triviaqa_{args.triviaqa_subset}_{args.triviaqa_split}.jsonl", subset=args.triviaqa_subset, split=args.triviaqa_split, limit=args.limit)
    print("wrote", out_dir / f"triviaqa_{args.triviaqa_subset}_{args.triviaqa_split}.jsonl", n)


if __name__ == "__main__":
    main()
