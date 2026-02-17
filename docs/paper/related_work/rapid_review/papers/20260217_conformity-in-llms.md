# Conformity in Large Language Models

- Year: 2024 (arXiv; ACL 2025 Main in comments)
- Venue: arXiv (ACL 2025 Main)
- Authors: Xiaochen Zhu, Caiqi Zhang, Tom Stafford, Nigel Collier, Andreas Vlachos
- URL: https://arxiv.org/abs/2410.12428
- BibTeX key (if we add it): Zhu2024ConformityLLMs
- Tags: conformity, social-influence, majority-pressure, uncertainty, mitigation

## One-sentence takeaway

LLMs systematically **shift their answers toward a (possibly wrong) majority**, and this conformity grows when the model is more **uncertain**, with simple prompt-based interventions reducing susceptibility.

## What problem does it solve?

- Quantifies “**conformity bias**” in LLMs: when shown a majority opinion/answer, models may align with it even if it is incorrect, which is risky for decision-support and information-seeking settings.
- Identifies when/why conformity is stronger (e.g., uncertainty) and proposes mitigations.

## What is the core method / protocol?

- Adapts classic psychology conformity experiments to LLMs:
  - Elicit an **initial answer** from the model.
  - Present **majority responses** (with different tones / naturalness) that may be correct or incorrect.
  - Measure whether the model **switches** to match the majority.
- Studies factors affecting conformity:
  - **Model type / training paradigm** (instruction-tuned vs others).
  - **Input characteristics**, including the *naturalness* / tone of the majority.
  - **Model uncertainty** (they report higher conformity when the model is less certain).
- Mitigation interventions (prompt-level):
  - **Devil’s Advocate**: push the model to consider counterarguments / the opposite stance.
  - **Question Distillation**: reframe/condense the question to reduce social/contextual contamination.

## What are the key metrics?

- Conformity rate: probability the model’s final answer matches the majority after exposure.
- Analyses conditioned on:
  - initial correctness vs incorrectness,
  - uncertainty level,
  - majority tone/naturalness,
  - model family / instruction tuning.

## What are the main results?

- All tested models exhibit **non-trivial conformity** across domains, even when their initial answer was correct.
- Conformity increases when the model is **more uncertain** in its own prediction.
- **Instruction-tuned** models are **less susceptible** to conformity (relative trend).
- Making the majority’s language more **natural / persuasive** increases conformity.
- Prompt-based interventions (Devil’s Advocate; Question Distillation) **reduce conformity** (mitigation evidence).

## How is this similar to GALILEO?

- Same broad failure mode family: **pressure-induced drift** in multi-turn settings (here: majority pressure / social influence).
- Reinforces the importance of:
  - separating *truth/evidence* from *social/contextual pressure*,
  - measuring how behavior changes **after a perturbation** (pre vs post exposure),
  - mitigation baselines that preserve capability while improving robustness.

## How is this different from GALILEO?

- Focus is majority-conformity setups rather than GALILEO’s targeted multi-turn robustness framing (e.g., time-to-failure / recovery dynamics).
- Emphasizes **uncertainty → conformity** correlations; does not (from the abstract-level view) foreground survival-style trajectory metrics or explicit recovery-after-flip objectives.
- Interventions are prompt-based “debiasing” steps, not evaluation protocols designed to distinguish **evidence-driven belief revision** from **pressure-driven drift**.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes explicit **control conditions** (e.g., evidence-present vs pressure-only) and **trajectory metrics** (time-to-flip, recovery), it can make a cleaner claim about drift vs revision than majority-only conformity.

## Where GALILEO is weaker / needs to improve

- If we do not include a “**group majority**” operator, we may miss an important and realistic social-pressure source (consensus cues are common in real tools/products).
- If we rely on confidence/logprob as uncertainty, this paper highlights uncertainty is behaviorally important—worth validating which uncertainty proxy works for our models.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a **majority-consensus pressure operator** (N synthetic agents / judges giving the same wrong suggestion) to test conformity-to-consensus vs single-user pressure.
- [ ] Report robustness conditioned on an uncertainty proxy (e.g., self-reported confidence; entropy/logprob if available) to test the “**more uncertain → more drift**” hypothesis.
- [ ] Include Devil’s-Advocate-style prompts as a lightweight mitigation baseline and compare against any GALILEO-specific interventions.

## Quotes / details to potentially cite

- “The conformity effect describes the tendency of individuals to align their responses with the majority.”
- “All tested models exhibit varying levels of conformity toward the majority… across different knowledge domains.”
- “LLMs are more likely to conform when they are more uncertain in their own prediction.”
- “Instruction-tuned models are less susceptible to conformity… increasing the naturalness of majority tones amplifies conformity.”
- “We propose two interventions, Devil's Advocate and Question Distillation, to mitigate conformity.”
