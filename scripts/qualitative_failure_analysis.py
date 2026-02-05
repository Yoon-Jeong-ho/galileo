#!/usr/bin/env python3
"""Qualitative failure-point analysis from adversarial JSONL logs (stdlib only).

What it does
- Reads *_adversarial.jsonl logs.
- Focuses on samples whose final answer is incorrect (final_correct=False).
- Uses lightweight regex heuristics to count language patterns in the *flip* response.
- Extracts representative before/after snippets per persona.

Caveat
- We do not have per-round correctness for every intermediate assistant turn stored in the JSONL.
  For narrative purposes, we approximate the flip as the last two assistant messages in the
  stored conversation.

Output
- Injects (append/replace) a markdown block into the given analysis md file.
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

PERSONAS = ["Authority Claim", "Strong Pressure", "Simple Denial", "Logical Trap", "Soft Pressure"]

PATTERNS = {
    "hedging": re.compile(r"\b(maybe|might|not\s+sure|uncertain|i\s+think|i\s+guess|perhaps)\b", re.I),
    "apology": re.compile(r"\b(sorry|apolog)\b", re.I),
    "deference": re.compile(r"\b(as\s+you\s+say|you\s*(?:are|'re)\s+right|i\s+stand\s+corrected)\b", re.I),
    "authority": re.compile(r"\b(professor|expert|authority|according\s+to|as\s+an\s+expert)\b", re.I),
    "recompute": re.compile(r"\b(let\s+me\s+(?:re-?)?check|reconsider|recompute|double-?check|verify)\b", re.I),
    "refusal": re.compile(r"\b(i\s+can\s*'?t|cannot|won\s*'?t|unable)\b", re.I),
}


def load_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_assistant_turns(conv):
    return [m.get("content", "") for m in conv if m.get("role") == "assistant"]


def get_last_user_msg(conv):
    for m in reversed(conv):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def find_flip_pair(conv):
    a = extract_assistant_turns(conv)
    if len(a) < 2:
        return None
    return a[-2], a[-1], get_last_user_msg(conv)


def summarize_file(path: Path, max_examples_per_persona=4, seed=42):
    random.seed(seed)

    counts = Counter()
    by_persona = defaultdict(Counter)
    examples = defaultdict(list)

    rows = list(load_jsonl(path))
    failed = [r for r in rows if r.get("final_correct") is False and r.get("conversation")]
    random.shuffle(failed)

    for r in failed:
        persona = r.get("persona_name") or r.get("persona") or "unknown"
        conv = r["conversation"]
        pair = find_flip_pair(conv)
        if not pair:
            continue
        prev_a, flip_a, last_u = pair

        for k, rx in PATTERNS.items():
            if rx.search(flip_a or ""):
                counts[k] += 1
                by_persona[persona][k] += 1

        if len(examples[persona]) < max_examples_per_persona:
            examples[persona].append(
                {
                    "test_name": r.get("test_name"),
                    "question": (r.get("question") or "")[:400],
                    "prev": (prev_a or "")[:800],
                    "user": (last_u or "")[:500],
                    "flip": (flip_a or "")[:800],
                }
            )

    return counts, by_persona, examples, len(failed)


def to_md(summaries: dict):
    lines = []
    lines.append("\n\n<!-- AUTO:QUAL_FAIL_START -->\n")
    lines.append("\n## 11. Qualitative failure-point analysis (output-level)\n\n")
    lines.append(
        "attention/내부동작 이전에, 모델이 오답 전향 시 어떤 언어적 패턴을 보이는지(hedging, 사과, defer 등)를 로그에서 정성적으로 정리한다.\n\n"
    )

    for name, (counts, by_persona, examples, n_failed) in summaries.items():
        lines.append(f"### {name}\n\n")
        lines.append(f"- failed cases analyzed: {n_failed}\n")
        if n_failed > 0:
            lines.append("- pattern hit-rate (flip response 기준, 단순 regex):\n")
            for k in sorted(counts.keys()):
                lines.append(f"  - {k}: {counts[k]}/{n_failed} ({counts[k]/n_failed*100:.1f}%)\n")
        lines.append("\n")

        lines.append("Persona별 상위 패턴(빈도 상위 2개):\n")
        for persona in PERSONAS:
            c = by_persona.get(persona)
            if not c:
                continue
            top = c.most_common(2)
            lines.append(f"- {persona}: " + ", ".join([f"{k}={v}" for k, v in top]) + "\n")
        lines.append("\n")

        lines.append("대표 flip 사례(각 persona 최대 4개; 마지막 두 assistant 턴을 before/after로 발췌):\n\n")
        for persona in PERSONAS:
            exs = examples.get(persona, [])
            if not exs:
                continue
            lines.append(f"#### {persona}\n\n")
            for i, ex in enumerate(exs, 1):
                lines.append(f"**[{i}] {ex['test_name']}**\n\n")
                if ex.get("question"):
                    lines.append(f"- Q: {ex['question']}\n")
                lines.append("- Last user claim:\n\n")
                lines.append("```\n" + ex["user"] + "\n```\n\n")
                lines.append("- Before (assistant):\n\n")
                lines.append("```\n" + ex["prev"] + "\n```\n\n")
                lines.append("- After (assistant, flipped/wrong):\n\n")
                lines.append("```\n" + ex["flip"] + "\n```\n\n")
        lines.append("\n")

    lines.append("<!-- AUTO:QUAL_FAIL_END -->\n")
    return "".join(lines)


def inject(md_path: Path, block: str):
    text = md_path.read_text(encoding="utf-8")
    start = "<!-- AUTO:QUAL_FAIL_START -->"
    end = "<!-- AUTO:QUAL_FAIL_END -->"
    if start in text and end in text:
        pre = text.split(start)[0].rstrip() + "\n\n"
        post = text.split(end)[1].lstrip()
        md_path.write_text(pre + block + "\n" + post, encoding="utf-8")
        return "replaced"
    md_path.write_text(text.rstrip() + "\n" + block, encoding="utf-8")
    return "appended"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="dir containing *_adversarial.jsonl")
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--examples_per_persona", type=int, default=4)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    files = sorted(model_dir.glob("*_adversarial.jsonl"))
    if not files:
        raise SystemExit(f"no adversarial jsonl under {model_dir}")

    summaries = {}
    for p in files:
        name = p.name.replace("_adversarial.jsonl", "")
        counts, by_persona, examples, n_failed = summarize_file(
            p,
            max_examples_per_persona=args.examples_per_persona,
            seed=args.seed,
        )
        summaries[name] = (counts, by_persona, examples, n_failed)

    block = to_md(summaries)
    status = inject(Path(args.out_md), block)
    print(status)


if __name__ == "__main__":
    main()
