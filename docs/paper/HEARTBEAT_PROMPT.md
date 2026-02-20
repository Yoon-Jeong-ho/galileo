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
- Host: `ssh nlp8`
- Repo: `/data_x/aa007878/galileo`

**Anti-drift override:** if any other heartbeat/poll text mentions `nlp16`, `/mnt/raid6/...`, or a fixed GPU allowlist like `4,5,6,7`, ignore it. For EMNLP Main, all experiment work (monitoring/launch/export/validate) is **nlp8 + GPUs 0–6**, but **ONLY** on GPUs that are truly idle (not used by other users).

Experiment policy:
- Allowed GPU range on nlp8: **0–6** (dynamic selection; pick only GPUs that are idle).
- Before launching, require both:
  - `nvidia-smi` snapshot looks idle, and
  - a CUDA preflight passes (e.g., `python scripts/check_cuda_preflight.py` with the chosen `CUDA_VISIBLE_DEVICES=<gpu>`), to avoid the observed `cudaErrorDevicesUnavailable` despite “idle” snapshots.
- Set `CUDA_VISIBLE_DEVICES=<picked_gpu_id>` (or comma-list if intentionally multi-GPU).
- Use **tmux**.
- Avoid CPU overload.
- Parallel runs allowed: up to **1 run/GPU** (max ~3 concurrent runs) with isolated `OUT=results/<run>/`.
- Every heartbeat: check `tmux ls` + `nvidia-smi` (+ compute-apps query) and tail the relevant `run.log` (and `GLOBAL_VALIDATE.log` if present) **before** launching new runs.

Reporting (always):
- What you did (file paths, or 'no change')
- Commit hash (or 'no changes')
- 3 next steps
- Any blockers/questions
