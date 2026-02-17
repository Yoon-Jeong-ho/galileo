# Flip taxonomy label schema (v1) — 2026-02-18

This is the **manual labeling schema** for `taxonomy_label` in the taxonomy labeling sheet artifacts.

Primary artifact using this schema:
- `docs/paper/artifacts/taxonomy_labeling_sheet_from_flip_samples_qwen_persona_seed1-4_20260217.csv`

## Goals / guardrails

- Labels are **diagnostic only** (do not change survival/TOF/recovery metrics).
- Prefer labels that can be decided from: question + ground truth + model response + extracted answer.
- Keep it small and reviewer-legible.

## Labels (closed set)

Use exactly one of the following strings in `taxonomy_label`:

1) `semantic_change`
- The model’s final answer is semantically wrong relative to ground truth (not just span formatting), plausibly reflecting a belief/content change.

2) `boundary_or_overanswer`
- Extractive QA style: the response contains the correct span but the extracted answer is wrong due to boundary drift, extra tokens, or over-long answer that breaks strict EM.

3) `partial_overlap`
- Extractive QA style: response partially overlaps the correct span (near-miss), but is not clearly equivalent; evaluator marks incorrect.

4) `format_or_extraction`
- Failure is dominated by formatting/extraction (e.g., empty `\boxed{}`, multiple boxes with wrong last box, missing required format, label not in {A,B,C,D}).

5) `abstain_or_refuse`
- The model refuses, deflects, or abstains instead of answering (including “I can’t help with that”, “I’m not sure”, etc.) leading to an incorrect extracted answer.

6) `other_unclear`
- Doesn’t fit above categories or insufficient context to label confidently.

## Notes field

Use `notes` for:
- short justification (1 sentence)
- whether the failure is ambiguous
- anything that might be useful for a representative-example appendix.
