#!/usr/bin/env python3
"""Write paper_exports/runner_metadata.json for a run (paper-ready requirement).

Why: paper_export.py writes metadata.json about the export, but we also require a
runner-side provenance record (GPU, TP, num_samples, max_model_len, max_tokens,
seed, conda env, model) for auditability and parity checks.

Stdlib only.

Usage:
  python3 scripts/write_runner_metadata.py \
    --paper_exports results/<run>/paper_exports \
    --model deepseek-ai/deepseek-llm-7b-chat \
    --seed 2 --gpu_list 2 --tp 1 \
    --num_samples 80 --max_model_len 4096 --max_tokens 2048 \
    --conda_env galileo
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper_exports", required=True, help="Path to paper_exports/ dir")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu_list", required=True, help="e.g., '2' or '1,2'")
    ap.add_argument("--tp", type=int, required=True, help="tensor_parallel_size")
    ap.add_argument("--num_samples", type=int, required=True)
    ap.add_argument("--max_model_len", type=int, required=True)
    ap.add_argument("--max_tokens", type=int, required=True)
    ap.add_argument("--conda_env", default="galileo")
    args = ap.parse_args()

    pe = Path(args.paper_exports)
    pe.mkdir(parents=True, exist_ok=True)

    out = pe / "runner_metadata.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gpu_list": args.gpu_list,
        "tensor_parallel_size": args.tp,
        "num_samples": args.num_samples,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "conda_env": args.conda_env,
        "model": args.model,
        "seed": args.seed,
    }

    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
