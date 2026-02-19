# Tonight Submission Freeze Plan (GALILEO)

Goal: stop scope creep, freeze the narrative, and produce a reviewer-auditable package tonight.

## Freeze policy (effective now)

- No new experiment launches unless there is a **claim-blocking gap**.
- No new model-family expansion tonight.
- Prioritize: **claim/evidence alignment → figure/table consistency → submission-ready draft flow**.

## Deliverables for tonight (must all be true)

1. **Claim-proof lock**
   - `docs/paper/CLAIM_EVIDENCE_MAP.md` fully aligned with current draft pointers.
   - Every Abstract/Intro headline claim has exactly one obvious proof pointer.

2. **Figure/Table pointer lock**
   - `docs/paper/PAPER_DRAFT_EN.md` figure/table mentions match filenames in `docs/paper/figures/`.
   - `docs/paper/FIGURE_CAPTIONS.md` terminology is consistent (Survival/TOF/Fail@1/Recovery).

3. **Table W wording lock**
   - Caption/body language explicitly states matched-subset control interpretation and weighted vs unweighted aggregation meaning.

4. **Related-work positioning lock**
   - Distinction vs SYCON/TRUTH DECAY/rebuttal-style evaluator setups is crisp and non-overlapping.

5. **Repro lock**
   - Regeneration commands for key artifacts/figures are listed and valid.

## Ordered execution (single-pass)

1) Update `CLAIM_EVIDENCE_MAP.md` (only mismatches).
2) Apply matching edits in `PAPER_DRAFT_EN.md`.
3) Final pass in `FIGURE_CAPTIONS.md` and Table-W wording.
4) Run citation + consistency checks.
5) Commit as one freeze commit.

## Done criteria

- No unresolved "TODO" markers in Abstract/Intro/Results pointers.
- A reviewer can trace every major claim to one figure/table and one artifact path without ambiguity.
- Draft sections no longer reference stale experiment hosts/policies.

## Out of scope tonight

- New seed runs.
- New cross-family additions.
- Non-essential prose expansion.
