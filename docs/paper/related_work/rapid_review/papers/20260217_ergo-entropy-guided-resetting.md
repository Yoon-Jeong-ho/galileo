# ERGO: Entropy-guided Resetting for Generation Optimization in Multi-turn Language Models

- Year: 2025
- Venue: UncertaiNLP 2025 (Workshop on Uncertainty Aware NLP)
- Authors: Haziq Mohammad Khalid; Athikash Jeyaganthan; Timothy Do; Yicheng Fu; Sean O’Brien; Vasu Sharma; Kevin Zhu
- URL: https://arxiv.org/abs/2510.14077
- BibTeX key (if we add it): ergo2025entropy
- Tags: multi-turn, uncertainty, entropy, prompt-consolidation, intervention, robustness

## One-sentence takeaway

ERGO monitors **next-token predictive entropy** during multi-turn generation and, when it spikes, **reconstructs/consolidates the prompt** to “realign” context—reporting sizable gains on incrementally revealed instruction tasks.

## What problem does it solve?

- Multi-turn LLM interactions degrade when instructions/info arrive incrementally (“model gets lost”), leading to lower accuracy and higher variability.
- The paper targets *when* and *how* to intervene during a conversation to prevent accumulated context noise from derailing performance.

## What is the core method / protocol?

- Compute **Shannon entropy** over the model’s next-token distribution at each turn as an internal uncertainty signal.
- Detect **sharp entropy spikes** (thresholding / change detection) as indicators of “misalignment” / drift.
- On spike, trigger an **adaptive prompt consolidation / reconstruction** step:
  - preserve key task-relevant elements,
  - discard accumulated clutter/noise,
  - continue the conversation with the consolidated context (“resetting”).

## What are the key metrics?

- Task performance on multi-turn tasks with incrementally revealed instructions.
- “Aptitude” (best/peak performance capability) and “unreliability” (variance/instability across runs/turns) as reported summary statistics.
- (Implied) timing/precision of reset triggers vs alternative heuristics/baselines.

## What are the main results?

- Reports **+56.6% average performance gain** over standard multi-turn baselines on incrementally revealed instruction settings.
- Reports **+24.7% aptitude** increase and **−35.3% unreliability** decrease.
- Main claim: treating uncertainty as a *temporal control signal* enables practical, low-cost, real-time interventions.

## How is this similar to GALILEO?

- Same broad failure mode: **multi-turn degradation** / drift over turns.
- Uses a **time-varying signal** to decide *when* to intervene (adjacent to GALILEO’s interest in trajectory/turn-level analysis and “time-to-failure” style thinking).

## How is this different from GALILEO?

- ERGO’s signal is **token-level predictive entropy** (requires access to next-token probabilities / logits), whereas GALILEO evaluations may target model-agnostic behavioral outcomes (and may include closed models without logits).
- Focus is on **context reconstruction/reset** for instruction-following-like tasks, not explicitly on **social pressure / persuasion** or disentangling *pressure-driven drift* vs *evidence-driven belief revision*.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides paired conditions (neutral vs pressure vs evidence) and explicit flip/recovery metrics, it can offer a clearer causal story than “entropy spike ⇒ reset”.
- GALILEO can remain **black-box compatible** by avoiding reliance on internal logprob access.

## Where GALILEO is weaker / needs to improve

- GALILEO may need a comparably simple **online intervention baseline**; ERGO is a good example of a minimal, easy-to-communicate controller.
- If GALILEO lacks an “online monitor” story (predict imminent failure), ERGO is a precedent for that framing.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an “online monitor” baseline: **entropy/uncertainty spike detector** (or a black-box proxy) + **context consolidation** intervention; compare to no-intervention and static summarization.
- [ ] When writing, cite ERGO as evidence that **uncertainty-aware interventions** can improve multi-turn reliability (but note dependency on logits).
- [ ] If applicable, evaluate whether a **reset trigger** correlates with GALILEO’s flip / drift events (do spikes precede failures?).

## Quotes / details to potentially cite

- Abstract: “ERGO … quantifies internal uncertainty via Shannon entropy over next token distributions and triggers adaptive prompt consolidation when a sharp spike in entropy is detected.”
- Abstract: “In multi-turn tasks with incrementally revealed instructions, ERGO yields a 56.6% average performance gain … increases aptitude … and decreases unreliability ….”
