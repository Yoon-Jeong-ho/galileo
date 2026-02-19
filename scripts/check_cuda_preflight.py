#!/usr/bin/env python3
"""Quick CUDA preflight for a single visible GPU.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 scripts/check_cuda_preflight.py

Exit 0 on success, non-zero on failure.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(f"[FAIL] torch import failed: {exc}")
        return 2

    if not torch.cuda.is_available():
        print("[FAIL] torch.cuda.is_available() == False")
        return 3

    count = torch.cuda.device_count()
    print(f"[INFO] visible_cuda_device_count={count}")
    if count < 1:
        print("[FAIL] no visible CUDA devices")
        return 4

    try:
        torch.cuda.set_device(0)
        x = torch.tensor([1.0], device="cuda")
        _ = float(x.item())
    except Exception as exc:
        print(f"[FAIL] cuda allocation failed: {exc}")
        return 5

    print("[OK] cuda preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
