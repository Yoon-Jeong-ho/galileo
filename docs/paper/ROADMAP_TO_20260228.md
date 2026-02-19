# GALILEO EMNLP Main — Roadmap to 2026-02-28

Updated: 2026-02-19 20:55 KST

## Objective
By **2026-02-28**, finish all paper-critical experiments and lock a reviewer-auditable package:
- Claims ↔ artifacts ↔ figures ↔ text all aligned
- `results_paper/` parity-valid (`[OK] runner_metadata parity`)
- Final draft is editable by YOON with clear TODO markers

## Hard constraints (SSOT)
- Remote SSOT: `ssh nlp8`, repo `/data_x/aa007878/galileo`
- GPU policy: use **only idle GPUs among 0–6** (not used by other users)
- Long runs in tmux; one run per GPU by default
- Every cited run must have paper-ready `paper_exports/` + validator OK

## Work packages (WPs)

### WP1 — Experiment closure (Tier-1 first)
1. Close remaining Tier-1 family gaps (if any) and unresolved ablations.
2. Only run Tier-2 (extra seeds/tasks) when CI/story is fragile.
3. For every completed run:
   - stage to `results_paper/<alias>/paper_exports`
   - run global parity validation
   - sync minimal local bundle only when needed for figures/tables

**Done baseline (already):** Llama-3.2-3B, Phi-3-mini, Phi-3.5-mini, Mistral-Nemo, Zephyr-7B, Qwen2.5-14B, DeepSeek-7B (seed1–2 level in artifacts/SSOT).

### WP2 — Figure/table freeze
1. Keep canonical cross-family figure set synchronized with artifact CSVs.
2. Enforce one canonical tag policy (mutable `20260219` vs bump-on-change).
3. Freeze Table W / recovery / TOF references and LaTeX pointers.

### WP3 — Claims-to-evidence hardening
1. Complete `CLAIM_EVIDENCE_MAP.md` sentence-level pointers for Abstract/Intro.
2. Remove any text claim without direct artifact/figure citation.
3. Add short “what this evidence does NOT claim” notes for reviewer robustness.

### WP4 — Writing for acceptance
1. Tighten contribution boundaries and novelty framing.
2. Keep related-work TOP10 synchronized with draft framing (not volume-driven).
3. Leave explicit user-editable TODO blocks in draft for final polishing.

## Deadline checkpoints
- **D-7 (2026-02-21):** all Tier-1 experiments and parity checks closed.
- **D-5 (2026-02-23):** figures/tables frozen and draft evidence map complete.
- **D-3 (2026-02-25):** full paper pass (consistency + reviewer-risk edits).
- **D-1 (2026-02-27):** final smoke checks (LaTeX, artifact pointers, reproducibility notes).
- **D-day (2026-02-28):** handoff-ready package for YOON final edits/submission.

## Heartbeat execution rule (from now)
Each heartbeat must produce exactly one primary deliverable from:
- experiment closure,
- artifact/figure lock,
- claims-evidence hardening,
- paper text quality improvement.

No-op heartbeats are not allowed unless blocked by idle-GPU unavailability; if blocked, do writing/dev lane work.
