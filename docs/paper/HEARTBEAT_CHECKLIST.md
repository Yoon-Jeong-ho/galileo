# Heartbeat Checklist (GALILEO EMNLP Main)

This is a **process guardrail** to prevent drift (wrong server/GPU policy, lane starvation, missing git hygiene, and missing continuity updates).

## A) Start of every heartbeat (2–3 minutes)

1) Read:
   - `docs/paper/STATUS.md` (rolling rewrite)
   - `docs/paper/HEARTBEAT_LOG.md` (append-only)
2) Decide **ONE primary lane** (must pick exactly one):
   - Experiments / Research / Development / Writing
3) Declare the deliverable *in one sentence* (what will be different in 10 minutes?).

**If lane = Experiments:** first confirm you can actually reach the box.
- Quick sanity: `ssh nlp16 'hostname; whoami'`
- If SSH is blocked (keys/agent/etc.), **do not burn the heartbeat** debugging infra unless explicitly prioritized—switch this heartbeat to Writing/Development and record the SSH blocker in the DM update.

## B) If lane = Experiments (nlp16; tiered)

Remote is:
- Host: `ssh nlp16`
- Repo: `/mnt/raid6/aa007878/galileo`
- GPUs policy: `CUDA_VISIBLE_DEVICES=4,5,6,7`

Tier selection:
- Default **Tier 1** (risk reducers): 1 new model family (seeds 1–2) OR 1 ablation that strengthens C2/C3.
- Tier 2 only with explicit rationale in `OUT/run.log` + `docs/paper/STATUS.md`.

Minimum checks **before launching anything**:
1) `tmux ls`
2) `nvidia-smi -i 4,5,6,7`
3) Tail latest run logs:
   - `tail -n 50 results/<run>/run.log`
   - `tail -n 50 results/<run>/GLOBAL_VALIDATE.log` (if present)

Launch discipline:
- Max **1 run/GPU** (4/5/6/7).
- Every run must have its own `OUT=results/<run>/`.
- A run is "paper-ready" only if `paper_exports/` + `metadata.json` + `runner_metadata.json` exist and validator prints `[OK]` + `[OK] runner_metadata parity`.

## C) If lane = Research (continuous)

Deliverable must be one of:
- Find 1 *new* closely related paper (bench/protocol/metric neighbor) and add a vault note stub + 2–3 bullets on why it matters.
- OR upgrade **one vault note** in `docs/paper/related_work/papers/*.md` to include protocol + metrics + (if available) 1–2 quantitative results.
- AND (whenever possible) add at least **one concrete citation sentence** to `docs/paper/PAPER_DRAFT_EN.md` (so research turns into paper strength).

## D) If lane = Development

Deliverable must be one of:
- New/updated script that improves auditability/reproducibility.
- Validator/tooling improvements with a minimal test (e.g., run against an existing `paper_exports/`).

## E) If lane = Writing

Deliverable must be one of:
- A revised paragraph/section in `docs/paper/PAPER_DRAFT_EN.md` with tighter positioning/claims.
- Updated Results text that matches tracked artifacts.

## F) Git hygiene (always)

- If any repo file under `/home/aa007/.openclaw/workspace/galileo` changed:
  1) `git status`
  2) `git add ...`
  3) `git commit -m "..."`
  4) `git push origin main`

No uncommitted changes should carry over across heartbeats unless explicitly noted in `STATUS.md`.

## G) End of every heartbeat (mandatory)

1) Update `docs/paper/STATUS.md` (rolling rewrite; remove stale next-steps)
2) Append one entry to `docs/paper/HEARTBEAT_LOG.md`:
   - what changed + links to commits/results
3) DM update must include:
   - What you did (file paths, or 'no change')
   - Commit hash (or 'no changes')
   - 3 next steps
   - Blockers/questions
