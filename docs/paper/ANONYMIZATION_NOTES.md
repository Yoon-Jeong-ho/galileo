# Anonymization notes (submission packaging)

This is a checklist+map of **infra-identifying strings** that must not leak into an anonymized submission/artifact bundle.

## 1) What to remove/replace

- Hostnames: `nlp8`, `nlp16`, etc.
- Absolute paths: `/mnt/raid6/...`, `/data_x/...`, home dirs
- Usernames / IPs: `aa007878@...`, `163.152.*`
- Any run directory names that encode internal machine/user info

Recommended replacement text in paper/repo-facing docs:
- `/mnt/raid6/aa007878/galileo` → `<REMOTE_REPO_ROOT>`
- `/data_x/aa007878/galileo` → `<REMOTE_REPO_ROOT>`
- `ssh nlp16` → `ssh <REMOTE_HOST>`

## 2) Where we currently have infra strings (internal-only docs)

The following files contain absolute paths/hostnames and should be either:
- kept internal-only (not in the anonymized bundle), or
- sanitized before packaging.

Flagged by grep:
- `docs/paper/PAPER_DRAFT_KO.md` (many `/mnt/raid6/...` and `/data_x/...` examples)
- `docs/paper/PAPER_RESULTS_ANALYSIS_KO.md` (results roots incl. `nlp8:/data_x/...`)
- `docs/paper/REMOTE_EXPERIMENTS_RUNBOOK.md` (explicit `nlp8` + repo path)
- `docs/paper/HEARTBEAT_PROMPT.md`, `docs/paper/HEARTBEAT_CHECKLIST.md`, `docs/paper/STATUS.md` (infra policy notes)
- `docs/paper/HEARTBEAT_LOG.md` (historical infra context)

Paper-facing EN draft: we have already removed server names from Results preface and Table W header, but still verify before export.

## 3) One-shot audit command (pre-submission)

```bash
grep -RIn --exclude-dir='__pycache__' --exclude='*.svg' --exclude='*.png' \
  -E "(/mnt/raid6/|/data_x/|nlp16|nlp8|aa007878@|163\\.152\\.|ssh nlp)" \
  docs/paper
```

## 4) Packaging guidance

- For an anonymized submission bundle, consider exporting only:
  - `docs/paper/PAPER_DRAFT_EN.md`
  - `docs/paper/FIGURE_CAPTIONS.md`
  - `docs/paper/figures/*.svg` (or generated PDFs)
  - `docs/paper/artifacts/*.csv`
  - scripts necessary to regenerate figures from artifacts
- Exclude internal process docs (heartbeat/runbooks/logs/KO notes) unless explicitly sanitized.
