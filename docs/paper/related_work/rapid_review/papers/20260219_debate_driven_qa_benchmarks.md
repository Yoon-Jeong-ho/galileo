# Pretraining on the Test Set Is No Longer All You Need: A Debate-Driven Approach to QA Benchmarks

- Year: 2025
- Venue: COLM 2025 (accepted; arXiv preprint)
- Authors: Linbo Cao; Jinman Zhao
- URL: https://arxiv.org/abs/2507.17747
- BibTeX key (if we add it): cao2025pretraining
- Tags: debate, evaluation, QA, benchmark-construction, contamination

## One-sentence takeaway

Convert existing QA questions into structured Pro-vs-Con multi-round debates with an answer-blind judge to make benchmarks harder and more contamination-resistant without costly new data collection.

## What problem does it solve?

- Standard QA benchmarks saturate quickly as models improve; creating new benchmarks is expensive.
- Data contamination / memorization can inflate scores when test items leak into training.
- Open-ended “chat” evaluations and judge-based evals can be subjective / inconsistent; they want a more standardized, scalable protocol.

## What is the core method / protocol?

- Take an existing QA dataset and form a **structured adversarial debate** per question:
  - **Pro** debater is given (and must defend) the official/ground-truth answer.
  - **Con** debater must propose an alternative answer and defend it.
  - A **judge model** (blind to the official answer) adjudicates based on argument quality.
- Key design choices highlighted in the intro:
  - Remove multiple-choice distractor options (to avoid forcing defense of obviously-wrong choices / alignment weirdness).
  - Use **multi-round argumentation** to increase difficulty and penalize shallow memorization.
  - Use **double round-robin / role reversal** to reduce positional bias (each model appears as Pro and Con across pairings).
- They instantiate a benchmark on a subset of **MMLU-Pro** questions and standardize protocols + reference models.

## What are the key metrics?

- Debate outcomes (win-rate) from the judge (pairwise model ranking / head-to-head results).
- Robustness checks against contamination: compare standard QA accuracy vs debate performance when a model is fine-tuned on test questions.
- Judge reliability: whether weaker judges can still discriminate stronger debaters (agreement / separability of rankings).

## What are the main results?

- Debate evaluation can **break apparent gains from test-set contamination**:
  - A Llama 3.1 model fine-tuned on the test questions jumps in standard accuracy (reported example: **50% → 82%**) but **performs worse in debates**.
- Even relatively weaker judge models can **reliably distinguish** stronger vs weaker debaters (suggesting the protocol scales).
- Overall claim: you can “recycle” existing QA items into harder evaluations at a fraction of the cost of new benchmark creation.

## How is this similar to GALILEO?

- Shares the general goal of **more robust evaluation** beyond single-shot QA accuracy and beyond simple judge prompts.
- Uses structured interaction (multi-turn, adversarial roles) to surface reasoning quality.

## How is this different from GALILEO?

- Their primary contribution is an **evaluation protocol + benchmark construction pipeline** for QA (not a training-time method).
- Debate is anchored to a known correct answer (Pro is given the official answer), whereas other multi-agent setups may not reveal ground truth to any agent.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides task-general or method-level guarantees beyond QA (e.g., across modalities / tasks), it may generalize more broadly than their QA-focused protocol.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently relies on single-shot accuracy or standard judge rubrics, this paper is a reminder that **contamination can create large apparent gains** that debate-style eval may expose.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add (or cite) a **debate-style evaluation** variant for any QA-style benchmark we report: Pro gets official answer; Con must craft an alternative; judge is answer-blind.
- [ ] Include a short discussion/citation on **contamination-resistance** via interaction-based evaluation (use their Llama 3.1 fine-tuning example as motivation).
- [ ] If we already use LLM judges, consider a small ablation: **judge strength vs ranking stability** (their claim: weaker judges still separate strong debaters).

## Quotes / details to potentially cite

- “...transforms any existing QA dataset into structured adversarial debates—where one model is given the official answer to defend, and another constructs and defends an alternative answer—adjudicated by a judge model blind to the correct solution.” (abstract)
- Contamination example: fine-tuning on test questions improves standard accuracy (**50% → 82%**) but hurts debate performance. (abstract)
- Code/benchmark pointer: https://github.com/l6cao/Debate-Driven-Evaluation
