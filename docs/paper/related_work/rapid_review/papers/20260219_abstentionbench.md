# AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions

- Year: 2025
- Venue: arXiv
- Authors: Mark Ibrahim; S. M. Ali Eslami; and collaborators (FAIR at Meta) (see arXiv for full author list)
- URL: https://arxiv.org/abs/2506.09038
- BibTeX key (if we add it): Ibrahim2025AbstentionBench
- Tags: abstention, unanswerable, uncertainty, robustness, calibration, reasoning-rlhf

## One-sentence takeaway

AbstentionBench shows that frontier LLMs (especially “reasoning” fine-tuned variants) often fail to abstain on unanswerable/underspecified/false-premise questions, with reasoning post-training *reducing* abstention performance (~24% average drop).

## What problem does it solve?

- Lack of a holistic, systematic benchmark for evaluating whether LLMs can *withhold* answers (refuse / ask clarifying questions / say “I don’t know”) when a query is ill-posed or unanswerable.
- Prior work tends to focus on narrower settings (e.g., hallucination/factuality or safety refusals), leaving many real-world “uncertainty” scenarios under-evaluated.

## What is the core method / protocol?

- Curate **AbstentionBench**, described as a large-scale benchmark spanning **~20 datasets** covering diverse “should abstain” scenarios, including:
  - unknown/unknowable answers
  - underspecified queries (missing necessary context)
  - false premises
  - subjective/ambiguous interpretation
  - outdated information
- Add “abstain” variants of standard reasoning-heavy benchmarks (named in the paper): **GSM8K-Abstain**, **GPQA-Abstain**, **MMLU-Abstain**, by constructing questions with missing context / underspecification.
- Evaluate **20 frontier LLMs**, including models with explicit reasoning post-training.
- Use **automatic scoring** of abstention behavior via an LLM judge (paper states “quality-verified LLM judge”) to scale evaluation.
- Study interventions:
  - comparing instruction-tuned vs reasoning fine-tuned models
  - varying reasoning token budgets (more “thinking”)
  - using a carefully crafted system prompt intended to encourage abstention

## What are the key metrics?

- “Abstention performance” (paper frames as whether the model appropriately abstains vs incorrectly answers) across datasets/scenarios.
- Scenario-wise breakdowns (e.g., unknown answers vs underspecification vs false premise).
- Comparisons between model families and post-training styles (instruction-tuned vs reasoning-tuned).

(Implementation detail: scoring is automated via an LLM judge; this is relevant for reproducibility/brittleness concerns.)

## What are the main results?

- **Abstention is unsolved**: frontier LLMs frequently answer when they should not.
- **Scaling helps little**: model scale has “almost no effect” on abstention performance (per intro).
- **Reasoning fine-tuning hurts abstention**:
  - reasoning-optimized variants show ~**24% average drop** in abstention compared to non-reasoning counterparts.
  - models may express uncertainty in the reasoning trace but still produce a definitive final answer.
- **Prompting can help but doesn’t solve it**: a carefully crafted system prompt boosts abstention, but the paper argues it does not address the underlying inability to reason about uncertainty.

## How is this similar to GALILEO?

- Shared theme: **robustness of LLM behavior under distribution shifts / adversarial or ill-posed interaction conditions**.
- Similar to GALILEO’s concern with “failure modes that look good superficially” (e.g., confident incorrect answers) and the need for **protocolized evaluation** rather than one-off anecdotes.
- The observed “reasoning makes it worse” effect is conceptually adjacent to GALILEO’s interest in **post-training side effects** (capability gains that worsen reliability dimensions).

## How is this different from GALILEO?

- Focus is specifically on **abstention / uncertainty handling** (refusal, clarification, “I don’t know”), not on GALILEO’s core target (multi-turn instability / drift vs evidence-driven change, depending on the exact GALILEO framing).
- Uses an **LLM judge** for abstention scoring; GALILEO may emphasize more extractive / mechanical metrics or different evaluation pipelines.
- Benchmark is across many static datasets/scenarios; GALILEO may be more interactional / multi-turn protocol driven.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses more **deterministic scoring** (string/structure checks, known ground truth, or protocol-level invariants), it may avoid some judge brittleness.
- If GALILEO explicitly separates *evidence-driven revision* vs *hallucinated filling-in*, it can complement AbstentionBench’s aggregate abstention scoring.

## Where GALILEO is weaker / needs to improve

- Consider adding a first-class “**abstention/clarification**” axis: not just whether the model changed, but whether it *should refuse or request missing info* before committing.
- Consider testing the “**reasoning trace vs final answer mismatch**” failure mode explicitly (uncertainty expressed mid-chain but confident final output).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a small GALILEO evaluation slice with **underspecified prompts** where the correct behavior is to ask a clarifying question (and score whether it does).
- [ ] Add an analysis plot: **reasoning-budget vs abstention/robustness**, checking whether more reasoning tokens increase “confident completion” failure.
- [ ] In related-work writing: cite AbstentionBench as evidence that **reasoning post-training can trade off with reliability** (abstention as a concrete case).

## Quotes / details to potentially cite

- “Abstention remains understudied, without a systematic evaluation framework for modern LLMs.”
- “Reasoning fine-tuning degrades abstention (by 24% on average)…”
- Benchmark composition: “20 diverse datasets… including questions with unknown answers, underspecification, false premises, subjective interpretations, and outdated information.”
- “Scaling models is of little use” (re: abstention).
