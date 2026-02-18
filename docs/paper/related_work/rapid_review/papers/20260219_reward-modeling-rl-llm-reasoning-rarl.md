# Reward Modeling for Reinforcement Learning-Based LLM Reasoning: Design, Challenges, and Evaluation

- Year: 2026
- Venue: arXiv
- Authors: Pei-Chi Pan; Yingbin Liang; Sen Lin
- URL: https://arxiv.org/abs/2602.09305
- BibTeX key (if we add it): pan2026reward
- Tags: reward-model, rlhf, rlaif, rlvr, reasoning, survey, evaluation, reward-hacking

## One-sentence takeaway

A survey/unification paper arguing reward modeling is the central lever for RL-tuned LLM reasoning, proposing a “Reasoning-Aligned Reinforcement Learning (RARL)” lens plus a taxonomy of reward types, reward hacking failures, and evaluation pitfalls.

## What problem does it solve?

- The literature on RL-based reasoning improvements (RLHF/RLAIF/RLVR/etc.) is fragmented across reward types (outcome vs process), judge models, rule-based/verifiable rewards, and “test-time scaling” uses of rewards; this makes it hard to reason about failure modes (hallucination, bias, distribution shift) and to evaluate claims.
- Reward design is treated as an implementation detail in many works, despite being the controlling signal that shapes what policies learn and how they generalize.

## What is the core method / protocol?

- Conceptual framework + taxonomy (not a new algorithm):
  - Defines **RARL** as “RL with reasoning-aware rewards” covering RLHF/RLAIF, preference optimization, LLM-as-a-judge, and **RL with verifiable rewards (RLVR)**.
  - Categorizes reward design into three broad paradigms (as described in the intro):
    - **Model-based rewards** (learned reward models / judges; includes dimensions like granularity and semantics).
    - **Rule-based rewards** (heuristics, format rules, verifiers, programmatic checks).
    - **Self-reward** (models judging / rewarding their own outputs).
- Organizes a discussion of **reward hacking** as a pervasive failure mode, connecting it to evaluator imperfections, over-optimization, credit assignment issues, and distributional bias.
- Connects reward design to broader system challenges:
  - inference-time scaling / planning-guided decoding,
  - hallucination and bias mitigation,
  - augmented reasoning (tools/RAG) where rewards can supervise tool-use trajectories,
  - benchmark construction and contamination.

## What are the key metrics?

- As a survey, it does not introduce a single metric; it discusses evaluation dimensions/risks:
  - correctness vs faithfulness of intermediate steps,
  - robustness / generalization under distribution shift,
  - reward-model reliability (biases like position/self-preference for LLM judges),
  - benchmark leakage / contamination,
  - susceptibility to reward hacking.

## What are the main results?

- Main “result” is a structured map of the space + argument:
  - reward design is a unifying abstraction linking many observed reasoning gains and failures;
  - reward hacking should be expected and treated systematically;
  - current benchmarks and reward-evaluation practices are vulnerable (esp. contamination and misalignment between reward and true reasoning quality).

## How is this similar to GALILEO?

- If GALILEO uses any reward- or evaluator-driven learning/selection (e.g., preference models, judges, verifiers, or structured scoring of reasoning/tool trajectories), this paper’s framing aligns: the reward/evaluator effectively *defines* the learned behavior.
- The paper’s emphasis on robustness (generalization, bias, hallucination) and on evaluator failure modes is directly relevant to any system relying on automatic scoring.

## How is this different from GALILEO?

- This is primarily a **survey/framework** paper; it does not propose a concrete end-to-end method to implement.
- Focus is on RL/reward modeling for **LLM reasoning fine-tuning and inference-time guidance**, not specifically on GALILEO’s particular architecture or objectives.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses **verifiable, task-grounded checks** (programmatic constraints, external validators) rather than opaque LLM judges, that can be a cleaner supervision signal than learned reward models.
- If GALILEO has a well-defined, reproducible evaluation protocol, that addresses one of the paper’s highlighted concerns.

## Where GALILEO is weaker / needs to improve

- Any reliance on LLM-as-a-judge scoring can inherit the paper’s highlighted biases (position bias, self-preference, contamination) unless explicitly mitigated.
- If GALILEO optimizes to a scalar score, it may be vulnerable to **reward hacking** (shortcut behaviors that look good to the scorer but are not truly better reasoning).

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, position GALILEO using the paper’s taxonomy: model-based vs rule-based vs self-reward; outcome vs process rewards; verifiable vs judge-based.
- [ ] Add an explicit “reward hacking / evaluator failures” subsection: list plausible shortcuts GALILEO could exploit and how we detect them.
- [ ] If using judges: add ablations for judge bias (swap judge model, randomize answer order to test position bias, calibration checks).
- [ ] Add contamination checks for reasoning benchmarks and for any judge-training data.

## Quotes / details to potentially cite

- Abstract-level framing (paraphrase): reward modeling is “not merely an implementation detail but a central architect of reasoning alignment,” shaping what is learned and how it generalizes.
- Introduces “Reasoning-Aligned Reinforcement Learning (RARL)” as a unifying framework for reward paradigms in multi-step reasoning.
