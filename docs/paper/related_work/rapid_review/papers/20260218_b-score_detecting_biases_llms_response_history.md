# B-score: Detecting biases in large language models using response history

- Year: 2025
- Venue: ICML 2025 (main track) (per arXiv)
- Authors: An Vo et al. (see arXiv page for full list)
- URL: https://arxiv.org/abs/2505.18545
- BibTeX key (if we add it): bscore2025vo
- Tags: bias, evaluation, multi-turn, response-history, self-correction

## One-sentence takeaway

Comparing an LLM’s single-turn vs multi-turn answer distributions (where the model sees its prior answers) yields a simple “B-score” signal that can detect biased/unstable behaviors and improve answer verification beyond verbalized confidence.

## What problem does it solve?

- Detecting and characterizing LLM “bias” when ground-truth labels or a known unbiased target distribution are unavailable.
- Disentangling different phenomena that look like bias in single-turn evaluations (true preference vs difficulty-driven errors vs training-data imbalance).

## What is the core method / protocol?

- Evaluate models by repeating the *same* question multiple times under two settings:
  - **Single-turn:** independent runs of the question.
  - **Multi-turn:** a conversation repeating the identical question each turn, so the model observes its prior answers (no “are you sure?” scaffolding).
- Use a question framework spanning **9 topics** and **4 categories / wordings**:
  - **Subjective** (asks for opinion)
  - **Random** (asks for an unbiased random choice)
  - **Easy objective**
  - **Hard objective**
- Define **B-score** per answer option *a* as the difference between its probability under single-turn vs multi-turn sampling (Δ between distributions). Intuition: answers that are over-produced in single-turn but “wash out” once the model sees its own history are flagged.
- Apply B-score as an auxiliary signal for **answer verification** (accept/reject LLM answers) on both the authors’ question set and standard QA benchmarks.

## What are the key metrics?

- B-score (distribution-difference signal between single-turn and multi-turn probabilities).
- Answer verification accuracy / correctness of accept-reject decisions (authors compare to verbalized confidence and to single-turn frequency baselines).

## What are the main results?

- Multi-turn response history can substantially *reduce apparent bias* for **Random** questions (e.g., repeated random-number prompts become close to uniform once the model sees its past answers), but does **not** necessarily remove **Subjective** preference (multi-turn can remain strongly skewed).
- B-score is reported to be effective across Subjective/Random/Easy/Hard categories.
- Using B-score improves answer-verification accuracy:
  - **+9.3** on the authors’ proposed questions (vs their baselines).
  - **+2.9** on common benchmarks (MMLU, HLE, CSQA) (vs their baselines).
- Verbalized confidence is a weaker indicator of bias/verification utility than B-score in their experiments.

## How is this similar to GALILEO?

- Uses **interaction history** as a diagnostic signal: the model’s own prior outputs become part of the evaluation context.
- Emphasizes **runtime** / post-hoc signals (no model finetuning needed) to detect problematic behavior.

## How is this different from GALILEO?

- Their multi-turn setting is deliberately minimal (repeat the same question) to isolate history effects; GALILEO (presumably) uses richer conversational context, tool use, or structured evaluation protocols.
- Focus is on **bias detection and verification** rather than (only) improving task performance.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has explicit task structure, ground-truth checks, or calibrated scoring, it may avoid conflating “distribution shift due to conversational memory” with genuine safety/fairness improvements.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently relies mainly on single-turn sampling, this paper suggests adding a **history-aware** diagnostic could reveal hidden instability/bias.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an ablation: single-turn vs multi-turn (repeat-question) for any GALILEO bias/safety probes; measure whether history reduces spurious mode-collapse answers.
- [ ] Consider a “B-score-style” feature: distribution change when the model sees its own recent outputs, as a flag for biased / brittle prompts.
- [ ] In related work: cite as evidence that multi-turn context can *reduce random-choice biases* but not subjective preference; and that distribution-difference can aid answer verification.

## Quotes / details to potentially cite

- Abstract claim: “LLMs are able to ‘de-bias’ themselves in a multi-turn conversation in response to questions that seek a Random, unbiased answer.”
- Abstract claim: “B-score … improves the verification accuracy … compared to using verbalized confidence scores or the frequency of single-turn answers alone.”
- Reported gains: “+9.3” on proposed questions; “+2.9” on MMLU/HLE/CSQA (as stated in the paper’s intro/abstract).
