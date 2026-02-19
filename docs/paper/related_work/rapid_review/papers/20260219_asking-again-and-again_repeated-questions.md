# Asking Again and Again: Exploring LLM Robustness to Repeated Questions

- Year: 2025 (arXiv v3; initially 2024)
- Venue: arXiv
- Authors: Sagi Shaier, Mario Sanz-Guerrero, Katharina von der Wense
- URL: https://arxiv.org/abs/2412.07923v3
- BibTeX key (if we add it): shaier2025asking
- Tags: repeated-questions, prompt-structure, robustness, drift-control, reading-comprehension

## One-sentence takeaway

Repeating the same question 3–5 times inside a *single prompt* produces at most small, statistically non-significant accuracy changes, suggesting modern LLM QA accuracy is fairly robust to redundant/repeated question phrasing.

## What problem does it solve?

- Tests a simple prompt heuristic (“repeat the question to make the model focus”) under controlled settings.
- Clarifies whether repetition-induced “attention” effects exist for extractive-style reading comprehension (RC) QA.

## What is the core method / protocol?

- Task setup: reading comprehension triples (q, c, a); input is (context + question) depending on setting.
- Datasets: SQuAD, HotPotQA, Natural Questions (NQ); sample 500 questions per dataset (cost constraint).
- Models (5): GPT-4o-mini (API), DeepSeek-V3 (API), and open-weight Llama-3.1-8B, Mistral-7B, Phi-4-14B.
- Repetition factor: repeat the question Qx1 vs Qx3 vs Qx5.
- Prompt configurations (4):
  - Open-book: context then repeated question.
  - Closed-book: repeated question only.
  - QCQ: repeated question, then context, then repeated question again.
  - Paraphrasing: append model-generated paraphrases of the question after the original question.
- Total settings: 3 (repetition) × 4 (configs) × 3 (datasets) × 5 (models) = 180; total ~90k prompts.

## What are the key metrics?

- Accuracy via substring match: any gold answer string appears in the model output.
- Statistical test over repetition levels: non-parametric Friedman test (Shapiro-Wilk indicates non-normality).

## What are the main results?

- Across models/datasets/configs, question repetition has **no statistically significant** effect on accuracy.
- Reported max gains are modest (up to ~+6% in some small-model closed-book cases), but not significant overall.
- Friedman test: statistic ≈ 0.7118, p ≈ 0.70 → no meaningful difference among Qx1/Qx3/Qx5.
- Paraphrasing can slightly hurt larger models in some settings (interpreted as noise).

## How is this similar to GALILEO?

- Supports GALILEO’s need to separate *mere repetition / re-asking* effects from true “pressure” effects.
- Provides evidence that redundant question presentation alone is not a strong driver of behavior change (at least for RC accuracy).

## How is this different from GALILEO?

- Single-turn (within-prompt) repetition, not multi-turn dialogue pressure.
- Measures QA accuracy, not truthfulness under adversarial social pressure, nor survival/TOF/recovery dynamics.
- No ground-truth “resistance to persuasion” framing; it’s prompt-structure robustness.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO explicitly measures *multi-turn* dynamics (survival curve, turn-of-failure, recovery@flip) under persona pressure vs neutral re-asking control.
- GALILEO focuses on settings where social pressure can cause incorrect flips even with a well-defined ground truth.

## Where GALILEO is weaker / needs to improve

- Could be helpful to cite/replicate a simple “within-prompt repetition” ablation to rule out trivial repetition artifacts in GALILEO prompts (if any repetition exists).

## Action items for GALILEO (experiments / method / writing)

- [ ] Related work paragraph: cite this as evidence that “repetition alone” (within a prompt) is typically not enough to substantially move accuracy, motivating why GALILEO’s *persona pressure* + multi-turn framing matters.
- [ ] (Optional) Add a small ablation: for a subset of tasks, repeat the current question within a turn (Qx3) and confirm survival/TOF results are unchanged.

## Quotes / details to potentially cite

- “Repeating questions within a single prompt … does not improve model performance significantly.”
- Setup detail: sample size 500 per dataset; 180 settings, ~90,000 questions.
- Statistical test detail: Friedman test p ≈ 0.70 (no significant differences across repetition levels).
