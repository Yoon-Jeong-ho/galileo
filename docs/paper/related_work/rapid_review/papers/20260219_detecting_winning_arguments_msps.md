# Detecting Winning Arguments with Large Language Models and Persuasion Strategies

- Year: 2026
- Venue: arXiv (cs.CL)
- Authors: Tiziano Labruna; Arkadiusz Modzelewski; Giorgio Satta; Giovanni Da San Martino
- URL: https://arxiv.org/abs/2601.10660
- BibTeX key (if we add it): Labruna2026DetectingWinningArguments
- Tags: persuasion, winning-arguments, strategy-aware-prompting, interpretability, feature-extraction

## One-sentence takeaway

A simple but effective recipe: prompt an LLM to score an argument along six persuasion-strategy dimensions, then aggregate (avg or a small MLP) to better predict which argument “wins” in ChangeMyView-style pairs.

## What problem does it solve?

- Predict which of two replies in an online debate is more persuasive ("winning argument" / delta-awarded in ChangeMyView), and more broadly predict persuasiveness outcomes in other persuasion datasets.
- Provide more interpretable/structured signals than monolithic “is this persuasive?” scoring.

## What is the core method / protocol?

- **MS-PS (Multi-Strategy Persuasion Scoring):** for each message, run a 2-step prompt for each of 6 persuasion strategies:
  - Step 1: strategy-focused analysis (is it present / how it contributes)
  - Step 2: produce a **1–10 persuasiveness score** grounded in that analysis
- Six strategies used (taxonomy from prior persuasion-techniques work):
  - Attack on reputation; Justification; Simplification; Distraction; Call (encouragement to act/think); Manipulative wording
- Two aggregations:
  - **MS-PS-AVG:** average the six scores; pick higher mean (ties broken via LLM-based paraphrase/rewriting + re-score).
  - **MS-PS-MLP:** build a feature vector per message = 6 scores + (mean, variance, entropy). For WA pairs, concatenate both messages’ vectors (18 dims) and train an MLP classifier.
- Motivation/engineering detail: they report **direct pairwise comparison prompts show positional bias**, so they prefer independent scoring.

## What are the key metrics?

- **WA (Winning Arguments):** accuracy (%) on test set for predicting the delta-awarded message in each pair.
- **Anthropic/Persuasion + Persuasion for Good:** regression error (reported as **RMSE**, lower is better).
- They also compare persuasion-strategy detection vs human labels via micro-F1 (SemEval persuasion-techniques dataset).

## What are the main results?

- **Winning Arguments (accuracy, Table 1):** MS-PS variants improve over independent-scoring baselines across 5 LLMs.
  - Example (test accuracy %):
    - Llama-3.1-8B: MS-PS-AVG 60.72, MS-PS-MLP 61.34
    - Gemma-3-12B: MS-PS-AVG 62.83, MS-PS-MLP 63.69
    - Gemini-1.5: MS-PS-AVG 60.72, MS-PS-MLP 63.07
    - Gemini-2: MS-PS-AVG 61.83, MS-PS-MLP 62.70
    - OpenAI-o3: MS-PS-AVG 60.59, MS-PS-MLP 64.53 (they note this improvement is statistically significant)
- **Strategy detection vs human labels (SemEval 2023 persuasion techniques, Table 2):** MS-PS scoring → thresholding yields higher micro-F1 than single-prompt classification (0.722 ± 0.035 vs 0.664 ± 0.030).
- **Anthropic/Persuasion + Persuasion for Good regression (Table 3):** MS-PS-MLP strongly outperforms two prompt baselines (reported RMSE):
  - Anthropic: Baseline-1 1.55, Baseline-2 1.44, **MS-PS-MLP 0.82**
  - Persuasion for Good: Baseline-1 152.15, Baseline-2 102.07, **MS-PS-MLP 4.63** (this number is surprisingly low relative to baselines; worth sanity-checking if we reuse/compare).
- They also create **TWA**: topic-annotated WA via BERTopic into 4 broad domains, and show topic-dependent accuracy variation (Table 4).

## How is this similar to GALILEO?

- Uses **structured decomposition** of a complex latent property (persuasiveness) into **interpretable sub-dimensions**, then aggregates for a final decision.
- Treats LLM outputs as **feature generators** feeding a smaller supervised model (MLP), aligning with hybrid “LLM-as-encoder / scorer” patterns.

## How is this different from GALILEO?

- Task is **persuasion / argument quality** rather than scientific-reasoning evaluation or our core GALILEO objectives (depending on current paper scope).
- Their decomposition is a **fixed taxonomy of persuasion techniques**, whereas GALILEO typically needs domain/task-adaptive criteria.
- Heavy reliance on **LLM prompting and API models** (including OpenAI-o3, Gemini), not an end-to-end trained system.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO emphasizes dataset/task generality or formal evaluation protocols, we can position it as less tied to a particular persuasion-technique taxonomy.
- If GALILEO uses more principled uncertainty/calibration, we can contrast with heuristic tie-breaking via paraphrase loops.

## Where GALILEO is weaker / needs to improve

- Strategy-aware prompting as **interpretable intermediate supervision** may be a useful baseline or component if GALILEO needs explainable scoring.
- Their explicit note about **positional bias** in pairwise comparisons is a good reminder: GALILEO comparisons should randomize order / use symmetric prompting.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a baseline where an LLM produces **multiple dimension scores + small classifier/regressor** on top, to separate “prompting gains” from “representation/aggregation gains”.
- [ ] In any pairwise evaluation setting, include an **order-swap / positional-bias** diagnostic.
- [ ] If relevant to our task: consider whether a **taxonomy-driven rubric** (fixed criteria) improves reliability and interpretability, and whether to learn the aggregation.

## Quotes / details to potentially cite

- MS-PS overview: per strategy, “generate an explanation … followed by a 1–10 persuasiveness score”; AVG vs MLP aggregation (Figure 1).
- Direct comparison suffers “strong and systematic positional bias” (Section 5.2).
- WA dataset construction: each pair is (delta-awarded vs non-awarded) from same CMV thread, matched for topical similarity (Section 3.1).
