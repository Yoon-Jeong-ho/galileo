# When Persuasion Overrides Truth in Multi-Agent LLM Debates: Introducing a Confidence-Weighted Persuasion Override Rate (CW-POR)

- Year: 2025
- Venue: arXiv
- Authors: Mahak Agarwal; Divyam Khanna
- URL: https://arxiv.org/abs/2504.00374
- BibTeX key (if we add it): agarwal2025cwpor
- Tags: persuasion, multi-agent, debate, truthfulness, calibration, confidence

## One-sentence takeaway

Single-turn “debates” where one agent is correct and the other is persuasively wrong can reliably fool an LLM judge; CW-POR measures not just how often the judge is misled, but how confidently it endorses the falsehood.

## What problem does it solve?

- Quantifies a specific failure mode in multi-agent LLM debate/evaluation settings: rhetorical persuasion overriding factual correctness.
- Highlights that “being wrong” is worse when the model is *confidently* wrong; proposes a metric capturing severity (confidence-weighted overrides).

## What is the core method / protocol?

- Single-turn, 3-role setup on TruthfulQA:
  - Neutral agent: given question + ground-truth answer, produces an objective explanation under a word budget v.
  - Persuasive agent: given question + an incorrect distractor, produces an emotionally/authoritatively persuasive defense of the false claim under the same word budget.
  - Judge: sees both answers in random A/B order, selects which is correct, gives a 1–5 self-rated confidence + a 1-sentence rationale.
- Sweep verbosity v from 30 to 300 words (step 30).
- Compute an additional *log-likelihood confidence* by scoring two versions of the judge prompt ending in “Final Answer: Answer A/B”, softmaxing the final-token logprobs.
- Proposed metric CW-POR: weight override events by confidence (they report using combined confidence = normalized rubric confidence × LLC).

## What are the key metrics?

- POR (Persuasion Override Rate): fraction of questions where judge selects the persuasive-but-incorrect answer.
- Rubric confidence: judge self-reported confidence 1–5.
- LLC (Log-Likelihood Confidence): internal preference strength between A vs B (0.5–1).
- CW-POR: confidence-weighted POR (they use combined confidence = normalized rubric × LLC).

## What are the main results?

- Across five open-source models (3B–14B), persuasive false answers can override truthful answers, often with high combined confidence.
- CW-POR varies by TruthfulQA category; some categories show high CW-POR even if they have fewer samples (wide CIs).
- Some models show *higher* CW-POR on “non-adversarial” questions than on adversarial ones, suggesting “innocuous” queries can be especially dangerous when packaged persuasively.
- Verbosity effect: many models show a dip (lowest CW-POR) around ~90–120 words; very short (30–60) and very long (200+) responses can increase persuasion overrides.

## How is this similar to GALILEO?

- Directly relevant if GALILEO uses (or evaluates with) multi-agent critique/debate or any “compare two candidate answers/plans and pick the best” mechanism.
- Emphasizes the need to track *confidence calibration* when selecting between competing hypotheses/trajectories.

## How is this different from GALILEO?

- Task is TruthfulQA-style factual QA with a deliberately incorrect persuasive agent; not a grounded planning/control setting.
- Single-turn only (no rebuttal / cross-examination), and judge shares architecture family with agents.
- Metric focuses on persuasion/truthfulness, not task success, safety constraints, or environment-grounding.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has access to environment feedback, tool outputs, or verifiable checks, it can ground decisions beyond rhetoric.
- Multi-step verification / ensemble checks could reduce one-shot persuasion overrides.

## Where GALILEO is weaker / needs to improve

- Any selector/arbiter LLM that chooses between candidate outputs (plans, explanations, retrieved evidence) may be vulnerable to “confident-sounding” but wrong candidates.
- If GALILEO relies on self-reported confidence or single-model judging, it may reproduce CW-POR-like failures.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an ablation where the *selector/judge* must choose between (i) a grounded candidate and (ii) a rhetorically persuasive but slightly incorrect candidate; report both error rate and confidence-weighted error.
- [ ] When using debate/critique, try (a) requiring citations/checkable evidence, (b) tool-based verification, or (c) multi-turn cross-examination to reduce persuasion-only wins.
- [ ] Consider logging both self-rated confidence and a proxy for internal preference (e.g., logprob margin) to detect overconfident wrong choices.
- [ ] If applicable, test verbosity constraints: mid-length explanations may be safer than extremely short/long ones.

## Quotes / details to potentially cite

- “We introduce the Confidence-Weighted Persuasion Override Rate (CW-POR), which captures not only how often the judge is deceived but also how strongly it believes the incorrect choice.” (Abstract)
- Verbosity sweep: v ∈ {30, 60, 90, …, 300} words.
- Combined confidence used for CW-POR: (normalized rubric confidence) × (log-likelihood confidence).
