#!/usr/bin/env python3
"""Lightweight progress check for a GALILEO results directory.

Motivation
- `run.log` can be buffered/silent; JSONL mtimes are the most reliable liveness signal.
- Root-level CSVs are the completion signal for paper exports.

Usage
  python3 scripts/check_run_progress.py --results_root results/<run_dir>

Exit codes
- 0: looks alive or complete
- 2: likely stalled (no JSONL mtime update past threshold and not complete)

Stdlib-only.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT_CSVS = [
    "initial_accuracy.csv",
    "adversarial_survival.csv",
    "recovery_accuracy.csv",
]


@dataclass(frozen=True)
class JsonlStat:
    path: Path
    mtime: float
    size: int


def _find_model_subdir(results_root: Path) -> Path | None:
    subdirs = [p for p in results_root.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    # Prefer a dir that actually contains jsonls.
    subdirs.sort(key=lambda p: sum(1 for _ in p.glob("*.jsonl")), reverse=True)
    return subdirs[0]


def _collect_jsonls(model_dir: Path) -> list[JsonlStat]:
    out: list[JsonlStat] = []
    for p in model_dir.glob("*.jsonl"):
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        out.append(JsonlStat(path=p, mtime=st.st_mtime, size=st.st_size))
    out.sort(key=lambda s: s.mtime)
    return out


def _fmt_age(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True, type=Path)
    ap.add_argument("--stall_minutes", type=float, default=10.0)
    ap.add_argument("--tail", type=int, default=8)
    args = ap.parse_args()

    rr: Path = args.results_root
    if not rr.exists() or not rr.is_dir():
        print(f"[FAIL] results_root not found: {rr}", file=sys.stderr)
        return 1

    now = time.time()

    present_csvs = {p.name for p in rr.glob("*.csv")}
    have_all_csvs = all(name in present_csvs for name in ROOT_CSVS)

    model_dir = _find_model_subdir(rr)
    if model_dir is None:
        print(f"[WARN] no model subdir under {rr}")
        print(f"complete_root_csvs={have_all_csvs} present_root_csvs={sorted(present_csvs)}")
        return 0 if have_all_csvs else 1

    jsonls = _collect_jsonls(model_dir)
    if not jsonls:
        print(f"[WARN] no jsonl files under {model_dir}")
        print(f"complete_root_csvs={have_all_csvs} present_root_csvs={sorted(present_csvs)}")
        return 0 if have_all_csvs else 1

    latest = jsonls[-1]
    age = now - latest.mtime

    print(f"results_root={rr}")
    print(f"model_dir={model_dir.name}")
    print(f"complete_root_csvs={have_all_csvs}")
    print(f"present_root_csvs={[n for n in ROOT_CSVS if (rr / n).exists()]}")
    print(f"jsonl_count={len(jsonls)}")
    print(f"latest_jsonl={latest.path.name} age={_fmt_age(age)} size={latest.size}")
    print("latest_jsonl_tail:")
    for s in jsonls[-args.tail :]:
        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(s.mtime))
        print(f"  - {ts}  {s.size:>10d}  {s.path.name}")

    if have_all_csvs:
        print("[OK] complete (root CSVs present)")
        return 0

    if age > args.stall_minutes * 60:
        print(f"[STALL?] no JSONL mtime update for {_fmt_age(age)} (> {args.stall_minutes}m)")
        return 2

    print("[OK] alive (recent JSONL activity)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
