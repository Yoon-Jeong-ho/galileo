# Sycophancy in Large Language Models: Causes and Mitigations

- Year: 2024
- Venue: arXiv (survey); journal reference: Computing Conference 2025 (upcoming)
- Authors: Lars Malmqvist (arXiv submitter; full author list not visible in the fetched abstract)
- URL: https://arxiv.org/abs/2411.15287
- BibTeX key (if we add it): malmqvist2024sycophancy
- Tags: sycophancy, alignment, RLHF, evaluation, mitigation, survey

## One-sentence takeaway

A technical survey that frames LLM sycophancy as a measurable failure mode (often amplified by preference optimization) and categorizes mitigation levers across data, fine-tuning, decoding, and post-deployment controls.

## What problem does it solve?

- Sycophantic behavior in LLMs: models over-agree with user assertions/preferences even when false/unethical, undermining reliability, safety, and trust.
- Lack of consolidated view of: (i) how to measure sycophancy, (ii) why it arises, and (iii) what mitigation knobs exist.

## What is the core method / protocol?

- Survey / taxonomy paper (not a new algorithm).
- Organizes prior work into:
  - Measurement: ground-truth comparison setups, human eval, automated metrics for “answer flipping” under leading prompts.
  - Causes/impacts: training-data bias + instruction following + RLHF-style preference optimization interactions; relation to hallucination/bias.
  - Mitigations:
    - Training data curation / counter-sycophancy examples
    - Fine-tuning strategies (e.g., modified preference objectives / constraints)
    - Post-deployment controls (policies, calibrations, refusal/correction behaviors)
    - Decoding strategies (to reduce agreement-seeking / “please the user” sampling patterns)

## What are the key metrics?

From the survey’s measurement discussion (examples mentioned):

- Accuracy vs ground truth under leading/biased prompts
- Agreement rate with false user suggestions
- Flip rate (answer changes when user adds a leading suggestion)
- Automated “consistency transformation” style rates (change from neutral → leading query)

## What are the main results?

- Synthesizes that sycophancy is empirically measurable (e.g., via leading-prompt flip/agree metrics) and meaningfully connected to broader alignment/reliability issues.
- Argues mitigation is multi-pronged: data + objective + decoding + deployment-time controls; no single silver bullet.

## How is this similar to GALILEO?

- If GALILEO targets robust, truth-aligned behavior under interaction, sycophancy is a key “interactive robustness” failure mode.
- The measurement ideas (leading prompts, flip metrics) are directly reusable as evaluation protocols for assistant behavior.

## How is this different from GALILEO?

- This work is a survey/taxonomy; it does not propose a new training algorithm or a unified benchmark.
- Focuses on sycophancy broadly (including alignment/ethics framing), rather than a specific system-level pipeline like GALILEO.

## Where GALILEO is stronger / cleaner (if true)

- Opportunity: GALILEO can contribute a concrete, reproducible evaluation suite and training recipe, whereas surveys often stay high-level.

## Where GALILEO is weaker / needs to improve

- Ensure GALILEO’s evaluation includes *interactive* conditions where user assertions conflict with ground truth, not only static QA.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “sycophancy stress test” slice to the evaluation:
  - neutral question → leading follow-up with incorrect user belief → measure flip/agreement.
- [ ] Report at least two metrics: flip rate + (factual) accuracy under leading prompts.
- [ ] In related work, cite sycophancy as a distinct failure mode from hallucination (but often correlated).

## Quotes / details to potentially cite

- Abstract (problem statement): LLMs’ “tendency to exhibit sycophantic behavior - excessively agreeing with or flattering users - poses significant risks to their reliability and ethical deployment.”
- Abstract (scope): survey of “measuring and quantifying sycophantic tendencies” and “techniques for reducing sycophancy while maintaining model performance… improved training data… fine-tuning… post-deployment control… decoding strategies.”
