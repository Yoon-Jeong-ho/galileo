# Heartbeat Prompt (GALILEO EMNLP Main loop)

Copy/paste the block below into Slack DM to trigger the 10-minute loop.

> Single source of truth: if this conflicts with other prompts, **use this one**.

---

You are running a 10-minute **GALILEO EMNLP Main loop**.

**Goal:** continuously progress on *all* paper-related work by rotating across:
1) **Paper experiments**
2) **Paper research**
3) **Paper development (tooling/reproducibility)**
4) **Paper writing**

Per heartbeat:
- Produce **ONE primary high-impact deliverable** (choose exactly one lane).
- You may do small supporting actions from other lanes only if they are truly small.
- Rotate lanes over time so **Experiments / Research / Writing** never stall.
- Experiments are **tiered**:
  - Tier 1: cross-family generalization (1 new model family) or 1 high-value ablation.
  - Tier 2: seed5+ or extra tasks only if needed for CI/story.

Local repo (paper/code):
- `/home/aa007/.openclaw/workspace/galileo`

Remote experiments:
- Host: `ssh nlp16`
- Repo: `/mnt/raid6/aa007878/galileo`

Experiment policy:
- Allowed GPUs: **4,5,6,7 only** (`CUDA_VISIBLE_DEVICES=4,5,6,7`).
- Use **tmux**.
- Avoid CPU overload.
- Parallel runs allowed: up to **1 run/GPU** (max 4 concurrent runs) with isolated `OUT=results/<run>/`.
- Every heartbeat: check `tmux ls` + `nvidia-smi -i 4,5,6,7` + tail `run.log` (and `GLOBAL_VALIDATE.log` if present) **before** launching new runs.

Reporting (always):
- What you did (file paths, or 'no change')
- Commit hash (or 'no changes')
- 3 next steps
- Any blockers/questions
