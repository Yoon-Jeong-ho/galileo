# Remote experiments runbook (nlp8)

This runbook is for the **Experiments lane** in the 10-min heartbeat loop.

## Policy (must follow)

- Remote host: `ssh nlp8`
- Repo: `/data_x/aa007878/galileo`
- GPUs: **4,5,6 only** (`CUDA_VISIBLE_DEVICES=4,5,6`)
- Always use `tmux` so runs survive disconnects.
- Avoid CPU overload: keep worker counts small; avoid multiple heavy runs at once.

**Anti-drift:** if any external prompt/poll text mentions `nlp16`, `/mnt/raid6/...`, or GPUs beyond 4/5/6, treat it as stale. This runbook is the SSOT.

## 0) Connectivity sanity (30s)

Run:

```bash
ssh nlp8 'hostname; whoami'
```

If this fails (key/agent issue), **do not spend the whole heartbeat** on infra unless explicitly prioritized.
Switch the heartbeat to Writing/Development and log the blocker.

If you need to restore access, see: `docs/paper/SSH_TROUBLESHOOT_REMOTE.md`.

## 1) Mandatory status checks (2–3 min)

```bash
ssh nlp8 '
  cd /data_x/aa007878/galileo || exit 1
  echo "== tmux =="; (tmux ls || true)
  echo "== GPU (4-6) ==";
  nvidia-smi -i 4,5,6 --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
  echo "== newest results ==";
  ls -1t results | head -n 8
'
```

Then tail logs for the newest run root:

```bash
ssh nlp8 '
  cd /data_x/aa007878/galileo || exit 1
  OUT=$(ls -1dt results/* | head -n 1)
  echo "OUT=$OUT"
  echo "== run.log =="; tail -n 80 "$OUT/run.log" || true
  echo "== GLOBAL_VALIDATE.log =="; tail -n 80 "$OUT/GLOBAL_VALIDATE.log" || true
'
```

## 2) Launch discipline

### Known model/hardware pitfalls (nlp8 RTX8000)

Keep this section up-to-date; it prevents wasting Tier‑1 budget on runs that will never become “paper-ready.”

- **Gemma2 + vLLM (Triton unified attention)** can fail on RTX8000 (cc7.5) with:
  - `OutOfResources: shared memory ... Required: 81920, Hardware limit: 65536`
  - and/or a strict check when `--max_model_len` exceeds the model's `max_position_embeddings`.

- **Falcon-7B (`tiiuae/falcon-7b-instruct`) + vLLM** can fail at engine init with:
  - `AttributeError: 'FalconConfig' object has no attribute 'rope_parameters'`
  - Interpretation: likely a **transformers ↔ vLLM compatibility** mismatch for this model.

- **Pythia-2.8B (`EleutherAI/pythia-2.8b-deduped`) max context**:
  - vLLM will reject `--max_model_len 4096` because the model-derived max is **2048**.
  - Fix: run with `--max_model_len 2048` (preferred) rather than setting `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`.

If you need an additional cross-family model, prefer a family already known to run cleanly on this hardware, or do a **smoke run** first (1 seed, small `--num_samples`) before committing a full sweep.

- Prefer **one run per GPU** (4/5/6).
- Each run must have its own `OUT=results/<run>/`.
- A run is paper-ready only if it emits:
  - `paper_exports/{survival_curve.csv,turn_of_failure.csv,flip_samples.csv}`
  - `paper_exports/metadata.json`
  - `paper_exports/runner_metadata.json`
  - validator prints `[OK] .../paper_exports` and `[OK] runner_metadata parity`

### Paper-only validation root (`results_paper/`)

We keep a **clean, paper-facing results root** that contains only the runs we actually cite.
This avoids legacy directories breaking global validation.

- Location: `results_paper/` (under the repo root on nlp8)
- Structure: `results_paper/<alias>/paper_exports -> ../../results/<run>/paper_exports` (symlink)
- Validation command:

```bash
ssh nlp8 '
  cd /data_x/aa007878/galileo || exit 1
  python3 scripts/validate_paper_exports.py --results_root results_paper --check_runner_parity \
    2>&1 | tee results_paper/GLOBAL_VALIDATE.log
'
```

Rule: **Only `results_paper/` needs to be PASS** for paper claims.

## 3) What to write into the DM update

Always include:
- which tmux sessions exist
- GPU util/mem (4–6)
- newest run root + whether logs show progress/errors
- if launching: exact command, OUT path, GPU id, and expected runtime
