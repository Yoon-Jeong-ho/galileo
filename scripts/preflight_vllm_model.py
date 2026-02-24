#!/usr/bin/env python3
"""Quick preflight to catch vLLM/transformers/model incompat *before* starting Tier-1 runs.

Usage (remote, recommended):
  CUDA_VISIBLE_DEVICES=5 conda run -n galileo python scripts/preflight_vllm_model.py \
    --model HuggingFaceH4/zephyr-7b-beta

It will:
- infer a conservative max_model_len from HF config,
- attempt to instantiate a vLLM LLM engine,
- print a single-line OK/FAIL summary.

This is intentionally lightweight: it does not run the full GALILEO protocol.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional


def _infer_max_len_from_hf_config(model_id: str) -> Optional[int]:
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        cands = []
        for k in [
            "max_position_embeddings",
            "n_positions",  # GPT-2 style
            "seq_length",
            "max_seq_len",
        ]:
            v = getattr(cfg, k, None)
            if isinstance(v, int) and v > 0:
                cands.append(v)

        # HF sometimes sets model_max_length to a huge sentinel; still prefer explicit embeddings.
        v = getattr(cfg, "model_max_length", None)
        if isinstance(v, int) and 0 < v < 1_000_000:
            cands.append(v)

        return min(cands) if cands else None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Optional override (will be capped to inferred length unless --allow_long_max_model_len).",
    )
    ap.add_argument(
        "--allow_long_max_model_len",
        action="store_true",
        help="Sets VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 (dangerous; use only if you know model uses RoPE safely).",
    )
    ap.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16"],
        default="auto",
        help=(
            "vLLM dtype override. Use float16 on GPUs without bf16 support (e.g., RTX8000 cc7.5). "
            "Default: auto (let vLLM decide)."
        ),
    )
    args = ap.parse_args()

    # NOTE: Many modern HF configs report very large context lengths (e.g., 128k).
    # For *preflight* we want a conservative setting that fits typical GPUs.
    inferred = _infer_max_len_from_hf_config(args.model)
    if inferred is None:
        inferred = 4096

    if args.max_model_len is None:
        # Default conservative cap to avoid OOM / KV cache init failures during preflight.
        max_len = min(inferred, 4096)
    else:
        max_len = args.max_model_len

    if not args.allow_long_max_model_len:
        # Never exceed inferred model limit unless explicitly permitted.
        max_len = min(max_len, inferred)

    if args.allow_long_max_model_len:
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    # Keep output minimal (cron-friendly)
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    try:
        from vllm import LLM

        llm_kwargs = dict(
            model=args.model,
            trust_remote_code=True,
            max_model_len=max_len,
            tensor_parallel_size=1,
            disable_log_stats=True,
        )
        if args.dtype != "auto":
            llm_kwargs["dtype"] = args.dtype

        _ = LLM(**llm_kwargs)
        print(
            f"OK model={args.model} max_model_len={max_len} inferred={inferred} dtype={args.dtype} CUDA_VISIBLE_DEVICES={cuda}"
        )
        return 0
    except Exception as e:
        print(
            f"FAIL model={args.model} max_model_len={max_len} inferred={inferred} dtype={args.dtype} CUDA_VISIBLE_DEVICES={cuda} err={type(e).__name__}: {str(e)[:220]}",
            file=sys.stdout,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
