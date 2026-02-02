#!/usr/bin/env python3
"""Create TriviaQA open-domain QA jsonl.

Output:
{"task":"qa","question":"...","answers":[aliases...]}

We use the rc variant because it is lighter, and take split=validation by default.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="rc")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out", default="/data_x/aa007878/galileo/data/triviaqa_rc.jsonl")
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    ds = load_dataset("trivia_qa", args.subset, split=args.split)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in ds:
            q = (ex.get("question") or "").strip()
            ans = ex.get("answer") or {}
            aliases = []
            if isinstance(ans, dict):
                if ans.get("value"):
                    aliases.append(str(ans["value"]))
                aliases.extend([str(a) for a in (ans.get("aliases") or [])])
            aliases = [a.strip() for a in aliases if a and a.strip()]
            # dedup
            seen=set(); uniq=[]
            for a in aliases:
                if a.lower() in seen:
                    continue
                seen.add(a.lower())
                uniq.append(a)

            if not q or not uniq:
                continue

            row = {"task": "qa", "question": q, "answers": uniq}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if args.limit > 0 and n >= args.limit:
                break

    print("wrote", out_path, "lines", n)


if __name__ == "__main__":
    main()
