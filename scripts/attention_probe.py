#!/usr/bin/env python3
"""Attention-based failure-point analysis (Transformers).

Idea
- Reconstruct the conversation prompt (chat template) up to the failure point.
- Run ONE forward pass with output_attentions=True.
- Compute lightweight summary stats (avoid storing/plotting huge matrices).

Why truncation?
- Full attention is O(L^2). For long contexts this is expensive.
- We therefore truncate to the last `--max_len` tokens. This is an approximation.

Metrics (per sample)
- last_layer_entropy_last_token: entropy of attention distribution for the last token
- last_layer_mass_to_last_user: attention mass from last token to the last user message span

Input
- Galileo *_adversarial.jsonl (lines contain `conversation` and `final_correct`).

Output
- CSV with per-sample metrics + group label (fail/survive).
"""

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def entropy(p: torch.Tensor, eps: float = 1e-12) -> float:
    p = p.clamp_min(eps)
    return float(-(p * p.log()).sum().item())


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_input(tokenizer, conv):
    try:
        return tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    except Exception:
        parts = []
        for m in conv:
            parts.append(f"{m['role'].upper()}: {m['content']}")
        return "\n\n".join(parts) + "\n\nASSISTANT:"


def token_span_of_last_user(tokenizer, conv):
    last_user_idx = None
    for i in range(len(conv) - 1, -1, -1):
        if conv[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        return None

    conv_prefix = [dict(m) for m in conv]
    conv_prefix[last_user_idx] = {"role": "user", "content": ""}

    full = build_input(tokenizer, conv)
    pref = build_input(tokenizer, conv_prefix)

    full_ids = tokenizer(full, add_special_tokens=False).input_ids
    pref_ids = tokenizer(pref, add_special_tokens=False).input_ids

    start = len(pref_ids)
    user_ids = tokenizer(conv[last_user_idx]["content"], add_special_tokens=False).input_ids
    end = start + len(user_ids)
    return start, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adversarial_jsonl", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--num_fail", type=int, default=30)
    ap.add_argument("--num_survive", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()

    fails, survs = [], []
    for r in load_jsonl(Path(args.adversarial_jsonl)):
        if not r.get("conversation"):
            continue
        if r.get("final_correct") is False:
            fails.append(r)
        elif r.get("final_correct") is True:
            survs.append(r)

    random.shuffle(fails)
    random.shuffle(survs)
    fails = fails[: args.num_fail]
    survs = survs[: args.num_survive]

    rows = []

    def run_one(r, label):
        conv = r["conversation"]
        text = build_input(tok, conv)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        span = token_span_of_last_user(tok, conv)

        if ids.shape[1] > args.max_len:
            cut = ids.shape[1] - args.max_len
            ids = ids[:, cut:]
            if span is not None:
                span = (max(0, span[0] - cut), max(0, span[1] - cut))

        with torch.no_grad():
            out = model(ids, output_attentions=True, use_cache=False)

        att = out.attentions[-1][0]  # heads, seq, seq
        last = att[:, -1, :]         # heads, seq
        p = last.mean(dim=0)         # seq

        ent = entropy(p)
        mass_claim = float("nan")
        if span is not None:
            s, e = span
            s = max(0, min(int(s), p.shape[0]))
            e = max(0, min(int(e), p.shape[0]))
            if e > s:
                mass_claim = float(p[s:e].sum().item())

        rows.append({
            "label": label,
            "persona": r.get("persona_name", r.get("persona")),
            "test_name": r.get("test_name"),
            "rounds_completed": r.get("rounds_completed"),
            "seq_len": ids.shape[1],
            "last_layer_entropy_last_token": f"{ent:.6f}",
            "last_layer_mass_to_last_user": "" if math.isnan(mass_claim) else f"{mass_claim:.6f}",
        })

    for r in fails:
        run_one(r, "fail")
    for r in survs:
        run_one(r, "survive")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "label","persona","test_name","rounds_completed","seq_len","last_layer_entropy_last_token","last_layer_mass_to_last_user"
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("wrote", out_csv, "rows", len(rows))


if __name__ == "__main__":
    main()
