import os
import random
from pathlib import Path

QA_DIR = Path(os.environ.get("QA_DIR", "/data_x/aa007878/galileo/data_qa_pilot"))
SEED = int(os.environ.get("SEED", "42"))
random.seed(SEED)

TARGETS = [
    ("arc_easy_val_100.jsonl", "arc_easy_val_1000.jsonl"),
    ("squad_val_100.jsonl", "squad_val_1000.jsonl"),
    ("triviaqa_rc_val_100.jsonl", "triviaqa_rc_val_1000.jsonl"),
]

for src_name, dst_name in TARGETS:
    src = QA_DIR / src_name
    dst = QA_DIR / dst_name
    if dst.exists():
        print("exists", dst)
        continue
    if not src.exists():
        print("missing src", src)
        continue

    lines = src.read_text(encoding="utf-8").splitlines(True)
    if not lines:
        print("empty src", src)
        continue

    out = []
    while len(out) < 1000:
        block = lines[:]
        random.shuffle(block)
        out.extend(block)
    out = out[:1000]

    dst.write_text("".join(out), encoding="utf-8")
    print("wrote", dst, "from", src)
