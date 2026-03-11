# 4-lane rotation plan (GALILEO EMNLP Main)

Goal: advance **paper writing / paper research / paper experiments / method+tooling** continuously without context drift.

## Hard rules

1) Each 10-min heartbeat picks **exactly one lane** as the primary deliverable.
2) Do **not** switch lanes mid-heartbeat.
3) Every heartbeat must leave a **verifiable artifact**:
   - a git commit (writing/research/method lanes), or
   - a launched/validated run root with logs (experiments lane).
4) Experiments policy (current): **nlp8 only**, GPUs **4/5/6 only**, use `tmux`.

## Lanes + minimum deliverable

### Lane A — Writing (paper text)
- Minimum deliverable:
  - Edit `docs/paper/PAPER_DRAFT_EN.md` (one paragraph/section rewrite)
  - Commit with message prefix `paper:`

### Lane B — Research (literature)
- Minimum deliverable:
  - Add/upgrade one note under `docs/paper/related_work/papers/` OR a dated note under `docs/paper/`
  - Add at least **one** concrete citation sentence into `PAPER_DRAFT_EN.md`
  - Commit with prefix `paper(research):`

### Lane C — Experiments (nlp8)
- Minimum deliverable:
  - Run monitoring: `tmux ls`, `nvidia-smi -i 4,5,6`, tail newest `run.log` (+ `GLOBAL_VALIDATE.log`)
  - If launching: one run per GPU, isolated `OUT=results/<run>/`, log rationale in `OUT/run.log`
  - Paper-ready target: `paper_exports/` + `metadata.json` + `runner_metadata.json` + validator OK

### Lane D — Method/Development (tooling/repro)
- Minimum deliverable:
  - One script/tooling/doc improvement that reduces reviewer risk or improves auditability
  - Commit with prefix `process:` or `dev:`

## Rotation schedule (repeat)

Recommended cycle (4 heartbeats = ~40 min):
1) Writing → 2) Research → 3) Experiments → 4) Method/Development

If a lane is blocked (e.g., SSH), you may skip it **once**, but record the blocker and return next cycle.

## DM update header (anti-drift)

Every DM update must begin with:
- `LANE = <Writing|Research|Experiments|Development>`
- `DELIVERABLE = <one concrete file/change/result>`
