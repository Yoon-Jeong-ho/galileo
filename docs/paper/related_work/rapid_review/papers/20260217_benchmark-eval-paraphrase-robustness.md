# On Robustness and Reliability of Benchmark-Based Evaluation of LLMs

- Year: 2025
- Venue: ECAI 2025
- Authors: Vincenzo Della Mea; Stefano Mizzaro; Kevin Roitero
- URL: https://arxiv.org/abs/2509.04013
- BibTeX key (if we add it): dellaMea2025benchmarkParaphraseRobustness (suggested)
- Tags: evaluation, robustness, reliability, benchmarks, paraphrasing, prompt-sensitivity

## One-sentence takeaway

Benchmark leaderboards can preserve *relative* model rankings under paraphrasing, while *absolute* accuracy drops substantially—so fixed-wording benchmarks can overstate real-world robustness.

## What problem does it solve?

- Standard benchmark evaluation uses a single, canonical wording per item, but real usage exhibits linguistic variability.
- The paper asks whether benchmark-based evaluation is (i) **reliable** as a comparative tool and (ii) **robust** in absolute performance when questions are paraphrased.

## What is the core method / protocol?

- Take questions from **six common MCQ-style benchmarks** (examples named: MMLU, ARC-C, HellaSwag; paper covers 6 total).
- Systematically generate multiple **paraphrases** for each question (controlled linguistic/syntactic rewordings).
- Evaluate **34 LLMs** (varied sizes/capabilities) on the original vs paraphrased variants.
- Analyze:
  - changes in **absolute effectiveness** (accuracy deltas)
  - stability of **model rankings** across paraphrase sets

## What are the key metrics?

- Accuracy / effectiveness on original benchmark items.
- Accuracy / effectiveness on paraphrased variants.
- Rank correlation / ranking stability across variants (the central “reliability” lens).

## What are the main results?

- **Rankings** of models are “relatively stable” under paraphrasing.
- **Absolute performance** drops “significantly” under paraphrased inputs.
- Interpretation: leaderboards may be reasonably comparative, but they can **overestimate generalization** to real-world rewordings.

## How is this similar to GALILEO?

- Shares the core evaluation concern that **surface-form perturbations** can induce large capability drops.
- Supports the broader argument that “single-score” evaluations can be misleading without **robustness-aware** protocols.

## How is this different from GALILEO?

- Focuses on **single-turn paraphrase robustness** for benchmark questions.
- Does not study **multi-turn dynamics**, time-to-failure, recovery, or social-pressure operators.
- The perturbation is semantic-preserving paraphrase, not adversarial persuasion/pressure.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO reports multi-turn trajectories (e.g., ToF/survival/PWC + recovery), it can diagnose *when/how* failures emerge, not just whether a paraphrase causes a drop.
- GALILEO’s pressure vs correction controls (if included) can separate “good updating” from “bad drift,” which paraphrase studies do not address.

## Where GALILEO is weaker / needs to improve

- GALILEO should ensure robustness claims are not overly tied to one phrasing of prompts/operators; paraphrase sensitivity can be a confound.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a small “**paraphrase robustness**” slice: paraphrase *our* base questions/prompts and report variance (even for a subset) to show results are not an artifact of a single wording.
- [ ] In writing: cite this as motivation for robustness-aware evaluation beyond fixed-wording benchmarks.

## Quotes / details to potentially cite

- “questions are presented in their original wording, thus in a fixed, standardized format.”
- “rankings remain relatively stable … absolute effectiveness scores change, and decline significantly.”
