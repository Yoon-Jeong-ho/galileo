# Mitigating Sycophancy in Decoder-Only Transformer Architectures: Synthetic Data Intervention

- Year: 2024
- Venue: arXiv
- Authors: Libo Wang
- URL: https://arxiv.org/abs/2411.10156
- BibTeX key (if we add it): wang2024mitigatingSycophancySDI
- Tags: sycophancy, mitigation, synthetic-data, rlhf, robustness

## One-sentence takeaway

A simple **synthetic-data intervention** finetuning recipe (generated with GPT-4o) reduces measured “sycophancy rate” on a small true/false QA set, though the protocol is single-turn and evaluation details are relatively light.

## What problem does it solve?

- LLMs trained with RLHF can exhibit **sycophancy** (over-agreeing / catering to user suggestions).
- The paper aims to **reduce sycophancy** for decoder-only transformers via a data-centric post-training approach.

## What is the core method / protocol?

- **Synthetic Data Intervention (SDI):** generate diversified synthetic training data intended to counteract sycophantic behavior.
- Use **GPT-4o** as the generator/assistant in the experimental pipeline.
- Train an SDI-updated model and compare against an “original untrained model” baseline.
- Provides a companion GitHub with dataset + code (per arXiv comment): https://github.com/brucewang123456789/GeniusTrail/tree/main/Synthetic%20Data%20Intervention

## What are the key metrics?

- Accuracy rate (on their evaluation set)
- “Sycophancy rate” (definition not fully specified in the abstract)

## What are the main results?

- SDI-trained model improves **accuracy** and reduces **sycophancy rate** vs the baseline on their 100-item true/false evaluation.

## How is this similar to GALILEO?

- Same broad target: **reducing pressure-driven agreement / sycophancy** introduced or amplified by post-training.
- Uses a **synthetic-data** lever, which is one plausible mitigation class we may want to compare against.

## How is this different from GALILEO?

- Appears **single-turn** and relatively small-scale (100 T/F questions), whereas GALILEO is focused on **multi-turn dynamics** (time-to-failure, recovery, oscillation).
- No clear separation (from the abstract) between **evidence-driven belief revision** vs **pressure-driven drift**.
- Metrics seem mostly endpoint rates (accuracy/sycophancy) without trajectory-level or survival-style reporting.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes paired neutral-vs-pressure controls and multi-turn trajectories, it can make **causal claims about pressure sensitivity** more clearly than a small single-turn rate comparison.

## Where GALILEO is weaker / needs to improve

- If we do not already include a straightforward **synthetic-data finetune mitigation**, this is a reminder that reviewers may expect at least one data-centric baseline.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a minimal **synthetic anti-sycophancy SFT** baseline (even if simple) to position GALILEO against common mitigation levers.
- [ ] In related work, mention SDI-style interventions as a “data-centric mitigation” bucket; contrast with GALILEO’s trajectory/controls focus.

## Quotes / details to potentially cite

- Abstract: “The experiment used **100 true and false questions**, and compared the performance of the model trained with synthetic data intervention and the original untrained model on multiple indicators.”
- Abstract: “The results show that the SDI training model … has significant effectiveness in reducing sycophancy phenomena.”
