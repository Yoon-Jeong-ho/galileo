# Robustness of large language models in moral judgements

- Year: 2025
- Venue: Royal Society Open Science
- Authors: Soyoung Oh; Vera Demberg
- URL: https://royalsocietypublishing.org/rsos/article/12/4/241229/235626/Robustness-of-large-language-models-in-moral
- BibTeX key (if we add it): oh2025robustness-moral-judgements
- Tags: robustness, moral-judgment, prompt-variance, moral-machine

## One-sentence takeaway

LLM “moral preferences” measured via Moral Machine-style forced-choice prompts are not robust: tiny prompt-format changes (even just relabelling options) can flip or invalidate choices, and unbalanced scenario generation can create spurious apparent preferences.

## What problem does it solve?

- Audits the *methodological validity* of eliciting moral preferences from LLMs using Moral Machine-style dilemmas and AMCE-style preference estimation.
- Identifies two major confounds in prior work: (i) prompt/label sensitivity; (ii) biased dataset generation that unevenly assigns attributes to options.

## What is the core method / protocol?

- Replicates Takemoto (Moral Machine scenario generation + prompting LLM to choose between two cases with a single-token label).
- Computes preference effects using AMCE (average marginal component effect) across dimensions (species, social value, gender, age, fitness, utilitarianism; plus interventionism, passenger-vs-pedestrian, lawfulness).
- Runs prompt perturbations that should be semantics-preserving:
  - Reverse label assignment (swap “Case 1/Case 2” labels)
  - Reverse content order under labels
  - Change labels from “Case 1/Case 2” to “(A)/(B)”
- Diagnoses scenario-generation bias and re-generates a more balanced dataset (paper claims this reduces spurious preferences).

## What are the key metrics?

- AMCE values per dimension (effect sizes interpreted as preference toward sparing attribute on right vs left).
- Valid response rate (responses that cleanly match a label vs invalid/abstaining).
- Distribution of label choices under prompt variants (e.g., tendency to pick “Case 1” regardless of content).

## What are the main results?

- Replication (LLaMa2-7B-chat, same hyperparams as Takemoto): results are “virtually identical” to prior report when using the original prompt format.
  - Setup details reported: temperature 0.6, top-p 0.9, seed 123, 50k samples; valid response rate 79.77%.
- Prompt sensitivity is severe:
  - Reversing labels/content changes selection rates markedly.
  - Switching labels to “(A)/(B)” causes the model to often output invalid answers or to strongly prefer “(B)”, independent of scenario content.
- Biased attribute distribution in generated scenarios can confound interpretation (apparent “preferences” can be artefacts of how attributes are distributed across options).
- Overall implication: conclusions about LLM moral alignment drawn from such evaluations are not stable unless robustness checks and balanced designs are enforced.

## How is this similar to GALILEO?

- Directly relevant to *robust evaluation* of LLM behavior under prompt/format perturbations.
- Highlights that “measured preferences” can be artefacts of protocol design—mirrors concerns in alignment/safety evaluation where small presentation changes yield large behavior shifts.

## How is this different from GALILEO?

- Focuses on moral-dilemma elicitation and AMCE-style preference estimation rather than GALILEO’s specific task setting (whatever GALILEO’s core evaluation target is).
- Emphasis is on *measurement validity* (prompt format + dataset generation bias) rather than building a new model or training method.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO already uses multi-prompt robustness, label randomization, and content-order randomization, it can claim stronger methodological hygiene than many “single-template” value evaluations.

## Where GALILEO is weaker / needs to improve

- If GALILEO relies on forced-choice labels (A/B, 1/2) or fixed templates, it may inherit the same label/position biases unless explicitly controlled.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “prompt invariance” section to evaluation: randomize labels, randomize option order, and include multiple semantically equivalent templates; report variance.
- [ ] Track and report invalid/abstention rates as first-class metrics (not just filter silently).
- [ ] Audit dataset generation for attribute-label correlations; enforce balance constraints.
- [ ] Consider eliciting *free-form rationales + structured choice* and analyze consistency across perturbations (with careful handling of verbosity/refusal artifacts).

## Quotes / details to potentially cite

- Abstract: “LLM responses are highly sensitive to prompt formulation variants as simple as changing ‘Case 1’ and ‘Case 2’ to ‘(A)’ and ‘(B)’.”
- Replication setup: “temperature 0.6… sampling rate 0.9… random seed of 123… sample size of 50,000… valid response rate 79.77%.”
- Prompt sensitivity finding: under “(A)/(B)” labels the model “almost always either gives an invalid answer or chooses option ‘(B)’.”

## Notes

- Accessible full text (open): https://pmc.ncbi.nlm.nih.gov/articles/PMC12015570/
