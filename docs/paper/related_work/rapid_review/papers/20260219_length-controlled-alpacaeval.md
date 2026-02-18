# Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators

- Year: 2024
- Venue: COLM 2024 (per arXiv comments)
- Authors: Yann Dubois; Balázs Galambosi; Percy Liang; Tatsunori B. Hashimoto
- URL: https://arxiv.org/abs/2404.04475
- BibTeX key (if we add it): dubois2024lengthcontrolledalpacaeval
- Tags: evaluation, llm-judges, bias, length-bias, causal-inference, regression

## One-sentence takeaway

A simple GLM-based “controlled direct effect” adjustment can largely remove AlpacaEval’s length bias and improve correlation with Chatbot Arena (Spearman 0.94 → 0.98) while reducing verbosity gameability.

## What problem does it solve?

- LLM-judge / auto-annotator based benchmarks (e.g., AlpacaEval) exhibit systematic biases that can be gamed; a prominent confounder is preference for longer outputs.
- The goal is a cheap, post-hoc debiasing method that (a) keeps the interpretability of a win-rate, (b) preserves basic symmetry/identity properties, and (c) improves robustness + agreement with human rankings.

## What is the core method / protocol?

- Treat the spurious correlate (length difference between candidate and baseline outputs) as an *undesirable mediator* in a causal graph.
- Fit a generalized linear model (logistic regression) to predict the LLM-judge preference from:
  - **Model term:** a per-model coefficient (relative to baseline).
  - **Length term:** a per-(model,baseline) coefficient times a normalized length-difference feature (they use a tanh of standardized length difference for diminishing returns).
  - **Instruction term:** an instruction-difficulty feature (shared across models; estimated via a separate regression step).
- Define the **length-controlled** score by taking the fitted model and predicting counterfactual preferences with **length difference set to zero**, i.e., removing the length term contribution.
- Use cross-validation + L2 regularization; additionally weakly regularize the length coefficient to mitigate truncation-based adversarial attacks.

## What are the key metrics?

- Spearman correlation of model rankings with LMSYS Chatbot Arena.
- “Length gameability”: sensitivity of scores to prompting a model to be verbose vs concise (reported as normalized std across verbosity prompts).
- Robustness to adversarial post-processing (example: truncation attack) and preservation of win-rate interpretability constraints.

## What are the main results?

- Correlation with Chatbot Arena increases from **0.94 to 0.98** after length control.
- Sensitivity to verbosity prompts substantially decreases:
  - Example in paper: baseline (gpt4_1106_preview) fluctuates roughly **22.9% → 64.3%** in raw AlpacaEval under verbosity prompting, but only **41.9% → 51.6%** under AlpacaEval-LC.
  - Normalized std across verbosity prompts drops **~25% → ~10%**.
- Regularization reduces vulnerability to truncation-based gaming (paper reports a large gain without regularization, much smaller with it).
- LC win-rates retain “nice” properties (baseline 50%; symmetry under swapping baseline/candidate), and can be used to predict win-rates against alternative baselines.

## How is this similar to GALILEO?

- Shared theme: **robust evaluation** in settings where naive automated metrics are exploitable or biased.
- The “control for spurious correlate via post-hoc adjustment” framing is relevant if GALILEO uses any learned / proxy scoring (including LLM-as-judge) in experiments or ablations.

## How is this different from GALILEO?

- This paper is specifically about **pairwise LLM-judge win-rate benchmarks** (AlpacaEval) and length bias; it does not propose new task datasets or capability measurements.
- The method is a lightweight statistical correction rather than a new model/system.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s evaluation is grounded in task-specific, objective metrics (or human evaluation with carefully controlled protocols), it may avoid some judge-specific artifacts entirely.

## Where GALILEO is weaker / needs to improve

- If GALILEO relies on LLM-judges for any headline results, it likely needs explicit **bias audits** (length, formatting, position, verbosity instructions) and potentially post-hoc corrections or study designs to mitigate them.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a small “metric gameability” check to any LLM-judge evaluation: re-run a subset with concise vs verbose prompting; report stability.
- [ ] If using pairwise LLM-judge outcomes, consider a GLM adjustment controlling for measurable mediators (length, list-ness, formatting features) and report both raw and controlled scores.
- [ ] In writing: explicitly discuss spurious correlates and interpretability constraints (symmetry/identity) for any corrected metrics.

## Quotes / details to potentially cite

- Framing: debiasing answers the counterfactual question “what would the preference be if the model and baseline had the same length?”
- Reported headline: Spearman correlation with Chatbot Arena improves **0.94 → 0.98**.
- Logistic regression structure includes model identity, length-difference (normalized + tanh), and instruction difficulty; LC score sets length difference to zero.
