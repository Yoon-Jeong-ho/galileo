# Rapid Related-Work Review (3-minute cadence)

Goal: continuously read papers adjacent to GALILEO (multi-turn robustness under pressure, sycophancy/persuasion, belief revision vs drift controls, stability across rounds), log structured notes, and periodically distill the **top-10 most relevant** works with:
- how they differ from GALILEO,
- where GALILEO is stronger,
- where GALILEO should improve (methods/experiments/claims).

This directory is **rolling / SSOT** for the rapid review process.

## Files (SSOT)

- `QUEUE.md`: what to read next (append as we discover candidates).
- `PROGRESS.md`: rolling dashboard (counts, tags coverage, next gaps).
- `TOP10.md`: rolling top-10 shortlist + compare/contrast.
- `papers/*.md`: one paper note per item (structured template).

## Policy

- One paper per run.
- Each paper note must include: citation, URL, what it claims, what we can reuse, what is missing, and actionable next steps for GALILEO.
- Keep notes **paper-facing** (avoid internal hostnames/paths).

## Automation reliability (cron hygiene)

- Avoid brittle inline edits like `python3 -c "..."` for updating `QUEUE.md` (quoting/newlines frequently break).
- Prefer dedicated helper scripts under `scripts/`.
  - For marking a queue entry done + linking a note by URL, use:
    - `python3 scripts/rapid_review_queue_mark_done.py --url <URL> --note <path> [--comment "..."]`

