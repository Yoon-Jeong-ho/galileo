# Benchmarking Adversarial Robustness to Bias Elicitation in Large Language Models: Scalable Automated Assessment with LLM-as-a-Judge

- Year: 2025
- Venue: Machine Learning (journal)
- Authors: Riccardo Cantini, Alessio Orsino, Marco Ruggiero, Domenico Talia
- URL: https://arxiv.org/abs/2504.07887 (also: https://doi.org/10.1007/s10994-025-06862-6)
- BibTeX key (if we add it): cantini2025benchmarking_bias_elicitation
- Tags: bias-elicitation, robustness, llm-as-judge, evaluation, jailbreak

## One-sentence takeaway

A scalable benchmark (CLEAR-Bias) + LLM-as-a-judge scoring pipeline shows that *no* evaluated LLM is robust to adversarial bias elicitation, with jailbreaks (incl. low-resource-language and refusal-suppression variants) reliably exposing safety failures.

## What problem does it solve?

- How to *systematically and scalably* measure LLM robustness to prompts that elicit biased / stereotyped content, including under adversarial “jailbreak” prompting.
- How to avoid fully manual, expensive annotation for large-scale safety evaluation by using an LLM-as-a-judge that is calibrated/selected against human labels.

## What is the core method / protocol?

- Build a prompt dataset, **CLEAR-Bias**, covering:
  - Bias dimensions: age, disability, ethnicity, gender, religion, sexual orientation, socioeconomic status
  - Intersectional categories (e.g., gender×ethnicity; ethnicity×SES; gender×sexual orientation)
  - Two task types: multiple-choice + sentence completion
  - Prompt augmentation with multiple jailbreak techniques (e.g., translation / obfuscation / prompt & prefix injection / refusal suppression / reward incentive / role-play), with multiple variants per attack.
- Choose a “judge” LLM by measuring agreement with human annotations on a curated set of prompt–response pairs; they report DeepSeek V3 as the most reliable judge in their study.
- Evaluate target LLMs by:
  - Running bias-probing prompts and scoring with the judge (safety score)
  - For categories that appear safe, stress-test with jailbreak variants to quantify robustness under adversarial elicitation.
- Define/track several evaluation measures (they frame robustness/fairness/safety; and also propose metrics for task misinterpretation in adversarial settings + jailbreak effectiveness).

## What are the key metrics?

- Judge-produced **safety scores** for model responses to bias-elicitation prompts.
- Robustness/fairness/safety measures (paper-defined; implemented over the CLEAR-Bias task suite).
- Jailbreak effectiveness / model vulnerability measures (how often attacks bypass safety filters).
- “Task misinterpretation” type measures in adversarial scenarios (to separate “refusal/safety” from “the model did something else”).

## What are the main results?

- **Uneven bias resilience** across categories; age, disability, and intersectional biases stand out as prominent vulnerabilities.
- **Scale is not sufficient**: some smaller models can be safer than larger ones, suggesting training/alignment choices and architecture matter.
- **No model is fully robust**: jailbreak strategies (notably low-resource language translation and refusal suppression) are broadly effective across model families.
- Slight safety improvements across successive generations; but **medical-domain fine-tuned models** can become *less* safe than general-purpose counterparts.

## How is this similar to GALILEO?

- Methodological neighbor for **scalable evaluation protocols**: automated scoring (“LLM-as-a-judge”) + structured prompt suites + adversarial stress testing.
- Shares the theme that robustness/safety can fail under **distribution shift / adversarial prompting** even if average-case behavior looks aligned.

## How is this different from GALILEO?

- Focuses on **bias elicitation** (fairness/safety) rather than belief dynamics / epistemic instability per se.
- Primary measurement is a judge-derived **safety score**, not time-to-failure / turn-level dynamics of belief revision.
- Tasks are mainly short-form completions/MCQ; less about long-horizon multi-turn conversational trajectories.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO emphasizes *long-horizon*, *multi-turn* instability and explicit separation of drift vs evidence-driven revision, it may provide a cleaner lens for “epistemic robustness” than safety-score snapshots.
- GALILEO can potentially avoid judge-model dependence by using more direct, behaviorally grounded metrics (depending on our design).

## Where GALILEO is weaker / needs to improve

- This paper is a good reminder that we may need a **systematic adversarial prompt taxonomy** (e.g., translation/obfuscation/refusal suppression) even when our core target isn’t “bias”.
- If GALILEO uses LLM-based evaluators, we should explicitly validate evaluator agreement / brittleness (they show judge selection matters).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an “adversarial prompting palette” section to GALILEO eval: include translation/obfuscation/refusal suppression style transforms as robustness stressors (even if the content is non-bias).
- [ ] If we use LLM-as-judge anywhere, include a short judge-selection/validation protocol (agreement with small human-labeled set; sensitivity analysis).
- [ ] Consider reporting a metric that separates **refusal** vs **task misinterpretation** vs **comply-but-wrong** (their misinterpretation metric is a useful pattern).

## Quotes / details to potentially cite

- CLEAR-Bias scale: “4,400 prompts” across seven bias dimensions + intersectional categories, across two task types, augmented with seven jailbreak techniques (each with multiple variants).
- Finding: “no model is fully robust to adversarial elicitation,” and low-resource-language + refusal-suppression jailbreaks are broadly effective.
