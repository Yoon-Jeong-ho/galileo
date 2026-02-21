#!/usr/bin/env python3
"""Fail-fast detector for vLLM generation-length cap pathologies in run logs.

We have observed vLLM batch-wise logic that caps `max_tokens` to satisfy
  prompt_tokens + max_tokens + reserve_tokens <= max_model_len.

When this forces `max_tokens` down to 1 token ("capped to 1"), the run is not
protocol-comparable for GALILEO Tier-1 evidence (answers are effectively forced
into 1-token outputs). This script scans run.log and exits non-zero if such
lines are found.

Usage:
  python3 scripts/check_runlog_for_token_caps.py results/<run>/run.log

Exit codes:
  0: no capped-to-1 lines found
  2: capped-to-1 lines found
  3: input file missing/unreadable
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path


CAP_RE = re.compile(r"requested max_tokens=(?P<req>\d+) capped to (?P<cap>\d+)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_log", type=Path)
    ap.add_argument("--max_show", type=int, default=12)
    args = ap.parse_args()

    p: Path = args.run_log
    if not p.exists():
        print(f"[ERROR] missing file: {p}", file=sys.stderr)
        return 3

    capped_to_1 = []
    caps = Counter()

    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, start=1):
                m = CAP_RE.search(line)
                if not m:
                    continue
                req = int(m.group("req"))
                cap = int(m.group("cap"))
                caps[(req, cap)] += 1
                if cap == 1:
                    capped_to_1.append((i, line.rstrip("\n")))
    except OSError as e:
        print(f"[ERROR] cannot read {p}: {e}", file=sys.stderr)
        return 3

    if not caps:
        print("[OK] no max_tokens cap warnings found")
        return 0

    print("[INFO] observed max_tokens cap warnings (req->cap : count):")
    for (req, cap), cnt in caps.most_common():
        print(f"  - {req}->{cap} : {cnt}")

    if capped_to_1:
        print(f"[FAIL] found {len(capped_to_1)} lines with capped-to-1 (non-comparable run)")
        for i, (lineno, s) in enumerate(capped_to_1[: args.max_show], start=1):
            print(f"  [{i:02d}] L{lineno}: {s}")
        if len(capped_to_1) > args.max_show:
            print(f"  ... ({len(capped_to_1) - args.max_show} more)")
        return 2

    print("[OK] cap warnings exist, but none capped to 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
