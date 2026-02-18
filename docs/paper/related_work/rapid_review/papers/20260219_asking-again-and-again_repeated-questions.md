# Asking Again and Again: Exploring LLM Robustness to Repeated Questions

- Year: 2025
- Venue: arXiv (preprint)
- Authors: Sagi Shaier; Mario Sanz-Guerrero; Katharina von der Wense
- URL: https://arxiv.org/html/2412.07923v3
- BibTeX key (if we add it): Shaier2025AskingAgain
- Tags: repeated-questions, robustness, drift, evaluation

## One-sentence takeaway

Repeating the same question 3–5x inside a prompt yields small, inconsistent accuracy changes (up to ~+6% in some cases) that are not statistically significant overall, suggesting modern LLM QA accuracy is fairly robust to naive question repetition.

## What problem does it solve?

- Tests a common prompt “folk remedy” (repeat the question to make the model focus) and quantifies whether it reliably improves QA accuracy.
- Helps characterize sensitivity/robustness of LLMs to redundant input structure in reading-comprehension style prompting.

## What is the core method / protocol?

- Evaluate 5 LLMs (GPT-4o-mini, DeepSeek-V3, Llama-3.1 8B, Mistral 7B, Phi-4 14B) on 3 RC datasets: SQuAD, HotPotQA, Natural Questions.
- Prompt variants:
  - Open-book: context then question repeated k times.
  - Closed-book: question repeated k times (no context).
  - QCQ: question (k times), context, question (k times).
  - Paraphrasing: add model-generated paraphrases appended to context.
- Repetition levels k ∈ {1, 3, 5}.
- Use substring-match accuracy; run non-parametric Friedman test for significance across repetition levels.

## What are the key metrics?

- Accuracy via substring matching against gold answers.
- Statistical significance across repetition conditions (Friedman test).

## What are the main results?

- Across models/datasets/configurations, question repetition does not significantly change accuracy.
- Some individual model/dataset slices show gains up to ~6% (notably for smaller models in closed-book), but the global effect is not significant.
- Friedman test across aggregated settings reports p ≈ 0.70 (no significant differences among k=1,3,5).
- Paraphrasing sometimes hurts larger models slightly, suggesting added “question variants” can introduce noise.

## How is this similar to GALILEO?

- Relevant as a robustness/sensitivity check: evaluates whether superficial prompt structure changes (redundancy) meaningfully alter QA performance.
- Provides a clean template for reporting “prompt perturbation” results with both slice-level deltas and an overall significance test.

## How is this different from GALILEO?

- Narrow intervention (literal repetition / paraphrase) and narrow task family (RC-style QA); does not study richer interaction protocols, long-horizon behavior, or safety/behavioral failure modes.
- Uses substring-match accuracy only; no calibration, uncertainty, or citation/faithfulness metrics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets more realistic interaction settings or multi-turn protocols, it can better reflect real user behavior than single-prompt repetition.
- If GALILEO uses stronger evaluation (e.g., faithfulness/citation correctness, abstention quality, cost/latency), it will give a more complete picture than accuracy-only substring match.

## Where GALILEO is weaker / needs to improve

- If GALILEO claims prompt-level robustness, it may need similarly simple perturbation baselines (like repetition) to demonstrate stability on “easy” prompt transformations.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “redundant question repetition” perturbation baseline (k=1/3/5) in any prompt-robustness section; report both per-slice deltas and an overall paired significance test.
- [ ] Consider separating effects by model scale and by context availability (open vs closed book), since small gains appear more often for smaller models in closed-book.
- [ ] If using paraphrasing/rewriting in GALILEO, include an ablation noting that paraphrase variants can add noise and degrade performance.

## Quotes / details to potentially cite

- Abstract-level claim: repetition can increase accuracy “by up to 6%” in some slices, but “we do not find the result statistically significant.”
- Statistical test: Friedman test statistic 0.7118 with p-value 0.70 (no significant differences across repetition levels).
