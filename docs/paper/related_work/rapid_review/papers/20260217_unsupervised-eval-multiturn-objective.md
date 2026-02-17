# Unsupervised Evaluation of Multi-Turn Objective-Driven Interactions

- Year: 2025 (arXiv; under review ICLR 2026)
- Venue: arXiv / ICLR 2026 (under review)
- Authors: Emi Soroka; Tanmay Chopra; Krish Desai; Sanjay Lall
- URL: https://arxiv.org/abs/2511.03047
- BibTeX key (if we add it): soroka2025unsupervised
- Tags: multi-turn; evaluation; objective-driven; unsupervised; goal labeling; goal completion; uncertainty

## One-sentence takeaway

Proposes a set of (largely) judge-free, unsupervised metrics for objective-driven multi-turn conversations: latent goal labeling (via LLM-guided clustering), goal completion (via end-token likelihood / completion modeling), and response uncertainty (via response-tree style branching).

## What problem does it solve?

- Evaluating objective-driven multi-turn human–LLM interactions when data are unlabeled and human annotation is expensive.
- Avoiding brittleness/unreliability of LLM-as-a-judge evaluation, especially under distribution shift (domain specialization, tool-use, long interactions).

## What is the core method / protocol?

- Define objective-driven interaction turns with a latent user goal.
- Three metrics (high level):
  1) **Goal labeling** via **LLM-guided clustering**:
     - Prompt an LLM to summarize each interaction’s goal into short text.
     - Embed summaries; run k-means with an overestimated k.
     - Prompt an LLM to *describe* each cluster using in-cluster and out-of-cluster examples (positive/negative sets).
     - Iteratively merge clusters by cosine similarity of embedded cluster descriptions, with an LLM deciding “merge or not” (the one explicit LLM decision point).
  2) **Interaction completeness / goal completion**:
     - Model “complete” interactions by appending a special end tag to the last response.
     - Train an adapted completion model (LoRA on a LLaMA-family completion model, per paper) so that complete conversations are in-distribution and incomplete are outliers.
     - Score completion via likelihood of emitting the end tag when prompted with the full interaction.
  3) **Uncertainty**:
     - Construct a **response tree** approximating the conditional distribution over responses for a prompt by generating branches above a probability threshold; use branching/entropy-like signals as uncertainty.

## What are the key metrics?

- Goal cluster assignment + cluster text labels (goal taxonomy discovered from data).
- Completion score based on P(end | conversation) under a completion model adapted to the target distribution.
- Uncertainty based on breadth/divergence of the response tree above a probability threshold.

## What are the main results?

- Claims validation on both open-domain and task-specific interaction datasets (details not fully captured in this rapid pass; see full paper for quantitative results and datasets).
- Qualitatively: provides scalable evaluation signals without needing reference “ideal” responses or heavy human labeling.

## How is this similar to GALILEO?

- Same overarching goal: **evaluate agent / assistant behavior in multi-turn, objective-driven interactions**, including completion and uncertainty.
- Emphasizes distribution shift (domain adaptation, tool use) as central to enterprise/agent evaluation.

## How is this different from GALILEO?

- Focuses on **unsupervised / unlabeled** evaluation signals derived from interaction distributions rather than rubric-based or judge-based scoring.
- Goal labeling is framed as **clustering + LLM-generated cluster descriptions** rather than predefined task labels.
- Completion metric relies on adapting a completion model with an explicit **end tag** and treating failures as outliers.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO already provides calibrated, task-grounded metrics (or human-audited rubrics), it may be easier to interpret than clustering-derived goal labels.
- If GALILEO avoids any LLM decision steps, it may be less sensitive to prompt/model drift than the “LLM decides merge?” step.

## Where GALILEO is weaker / needs to improve

- If GALILEO relies heavily on LLM judges or labeled data, this paper highlights a path to **reduce labeling/judge dependence** and to incorporate distributional signals.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding an **unsupervised “completion likelihood”** baseline: train/fit a small adapter (or lightweight model) to emit an end-of-interaction marker on historical “successful” runs; use as an anomaly detector for incomplete sessions.
- [ ] Consider a **goal taxonomy discovery** appendix/section: summarize interactions → embed → cluster → label clusters, as an exploratory analysis tool for error buckets.
- [ ] Compare uncertainty metrics: logprob/perplexity vs response-tree/semantic-branching uncertainty for multi-turn settings.

## Quotes / details to potentially cite

- “We introduce the first set of unsupervised metrics for objective-driven interactions…” (Abstract)
- Metrics target: “labeling user goals, measuring goal completion, and quantifying LLM uncertainty…” (Abstract)
- Assumption called out: majority of interactions are successful/complete, so failures are rare and hard to detect by inspection (Methodology assumptions).
