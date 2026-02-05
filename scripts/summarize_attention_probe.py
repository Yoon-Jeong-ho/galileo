#!/usr/bin/env python3
"""Summarize attention_probe CSVs (stdlib only).

Computes per-group mean/std for:
- last_layer_entropy_last_token
- last_layer_mass_to_last_user

Also reports delta (fail - survive).
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def mean_std(vals):
    vals = [float(v) for v in vals if v != "" and v is not None]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def read_rows(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    blocks = []
    blocks.append("\n\n<!-- AUTO:ATTENTION_PROBE_START -->\n")
    blocks.append("\n## 10. Attention probe (single-forward, truncated)\n\n")
    blocks.append("아래는 실험 로그에서 대화 입력을 재구성한 뒤, Transformers로 `output_attentions=True` 단일 forward를 수행하여 attention을 요약한 결과이다.\n\n")
    blocks.append("주의: attention은 O(L^2)이므로 마지막 256 tokens로 truncate한 근사치이며, *메커니즘 힌트*를 제공하는 용도이다.\n\n")

    for csv_path in args.csv:
        p = Path(csv_path)
        rows = read_rows(p)
        by = defaultdict(lambda: defaultdict(list))  # group -> metric -> vals
        for r in rows:
            g = r["label"]
            by[g]["entropy"].append(r["last_layer_entropy_last_token"])
            by[g]["mass"].append(r.get("last_layer_mass_to_last_user", ""))

        ent_f_m, ent_f_s = mean_std(by["fail"]["entropy"])
        ent_s_m, ent_s_s = mean_std(by["survive"]["entropy"])
        m_f_m, m_f_s = mean_std(by["fail"]["mass"])
        m_s_m, m_s_s = mean_std(by["survive"]["mass"])

        blocks.append(f"### {p.name}\n\n")
        blocks.append(f"- N(fail)={len(by[fail][entropy])}, N(survive)={len(by[survive][entropy])}\n")
        if ent_f_m is not None and ent_s_m is not None:
            blocks.append(f"- Entropy(last token, last layer): fail {ent_f_m:.3f}±{ent_f_s:.3f} vs survive {ent_s_m:.3f}±{ent_s_s:.3f} (Δ={ent_f_m-ent_s_m:+.3f})\n")
        if m_f_m is not None and m_s_m is not None:
            blocks.append(f"- Mass(to last user span): fail {m_f_m:.3f}±{m_f_s:.3f} vs survive {m_s_m:.3f}±{m_s_s:.3f} (Δ={m_f_m-m_s_m:+.3f})\n")
        blocks.append("\n")

    blocks.append("<!-- AUTO:ATTENTION_PROBE_END -->\n")

    out_md = Path(args.out_md)
    text = out_md.read_text(encoding="utf-8")
    start = "<!-- AUTO:ATTENTION_PROBE_START -->"
    end = "<!-- AUTO:ATTENTION_PROBE_END -->"
    if start in text and end in text:
        pre = text.split(start)[0].rstrip() + "\n\n"
        post = text.split(end)[1].lstrip()
        out_md.write_text(pre + "".join(blocks) + "\n" + post, encoding="utf-8")
        print("replaced")
    else:
        out_md.write_text(text.rstrip() + "\n" + "".join(blocks), encoding="utf-8")
        print("appended")


if __name__ == "__main__":
    main()
