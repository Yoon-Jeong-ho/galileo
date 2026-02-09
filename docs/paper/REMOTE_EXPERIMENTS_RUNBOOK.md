# Remote experiments runbook (nlp16)

This runbook is for the **Experiments lane** in the 10-min heartbeat loop.

## Policy (must follow)

- Remote host: `ssh nlp16`
- Repo: `/mnt/raid6/aa007878/galileo`
- GPUs: **4,5,6,7 only** (`CUDA_VISIBLE_DEVICES=4,5,6,7`)
- Always use `tmux` so runs survive disconnects.
- Avoid CPU overload: keep worker counts small; avoid multiple heavy runs at once.

## 0) Connectivity sanity (30s)

Run:

```bash
ssh nlp16 'hostname; whoami'
```

If this fails (key/agent issue), **do not spend the whole heartbeat** on infra unless explicitly prioritized.
Switch the heartbeat to Writing/Development and log the blocker.

## 1) Mandatory status checks (2–3 min)

```bash
ssh nlp16 '
  cd /mnt/raid6/aa007878/galileo || exit 1
  echo "== tmux =="; (tmux ls || true)
  echo "== GPU (4-7) ==";
  nvidia-smi -i 4,5,6,7 --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
  echo "== newest results ==";
  ls -1t results | head -n 8
'
```

Then tail logs for the newest run root:

```bash
ssh nlp16 '
  cd /mnt/raid6/aa007878/galileo || exit 1
  OUT=$(ls -1dt results/* | head -n 1)
  echo "OUT=$OUT"
  echo "== run.log =="; tail -n 80 "$OUT/run.log" || true
  echo "== GLOBAL_VALIDATE.log =="; tail -n 80 "$OUT/GLOBAL_VALIDATE.log" || true
'
```

## 2) Launch discipline

- Prefer **one run per GPU** (4/5/6/7).
- Each run must have its own `OUT=results/<run>/`.
- A run is paper-ready only if it emits:
  - `paper_exports/{survival_curve.csv,turn_of_failure.csv,flip_samples.csv}`
  - `paper_exports/metadata.json`
  - `paper_exports/runner_metadata.json`
  - validator prints `[OK] .../paper_exports` and `[OK] runner_metadata parity`

## 3) What to write into the DM update

Always include:
- which tmux sessions exist
- GPU util/mem (4–7)
- newest run root + whether logs show progress/errors
- if launching: exact command, OUT path, GPU id, and expected runtime
