# Self-Improvement of Language Models by Post-Training on Multi-Agent Debate

- Year: 2025 (v3 updated 2026-01)
- Venue: arXiv
- Authors: Ankur Samanta; Akshayaa Magesh; Runzhe Wu; Ayush Jain; Youliang Yu; Daniel Jiang; Boris Vidolov; Paul Sajda; Yonathan Efroni; Kaveh Hassani
- URL: https://arxiv.org/abs/2509.15172
- BibTeX key (if we add it): samanta2025maca (suggested)
- Tags: multi-agent, debate, self-improvement, post-training, RL, preference-learning, self-consistency

## One-sentence takeaway

Multi-Agent Consensus Alignment (MACA) uses RL preference learning over multi-round debate traces (including majority/minority reasoning) to post-train LMs to better exploit debate and improve single-model accuracy and self-consistency.

## What problem does it solve?

- How to get a training signal “stronger than the model” for self-improvement without external supervision.
- How to leverage multi-agent debate traces more effectively than one-shot majority voting or simple consensus rewards.

## What is the core method / protocol?

- Multi-agent debate setting: multiple model instances exchange reasoning over multiple rounds.
- MACA: post-training with RL where the learning signal is based on preferences over *full reasoning traces* from the debate.
- Key modeling choice (per abstract): learn to differentiate between majority vs minority reasoning traces; this is claimed to outperform:
  - binary “consensus reward” style objectives
  - SFT-based approaches that imitate debate outputs

## What are the key metrics?

- “Utilization of multi-agent debate setting” (reported as an improvement on MATH).
- Standard benchmark accuracy for individual models: MathQA.
- Self-consistency (reported on GSM8K).
- Generalization to unseen benchmarks: GPQA, CommonsenseQA.

## What are the main results?

From the abstract:

- Better at utilizing debate: +26.87% on MATH.
- Individually more accurate: +21.51% on MathQA.
- More self-consistent: +27.6% on GSM8K.
- Generalization: +16.3% on GPQA; +11.6% on CommonsenseQA.

## How is this similar to GALILEO?

- Uses *multi-round interaction* (debate) to reduce inconsistency and improve robustness-like properties (self-consistency).
- Motivates that aggregating across trajectories (majority/minority reasoning) can produce stronger signals than single-sample behavior—conceptually adjacent to our focus on multi-turn instability and trajectory-level evaluation.

## How is this different from GALILEO?

- Primarily a *training/post-training* paper (RL + preference learning), whereas GALILEO is centered on *evaluation* of multi-turn drift/instability under pressure/manipulation.
- “Multi-agent debate” is cooperative model–model interaction, not user–model conversational pressure with truth/stance drift.
- Focuses on math/QA benchmarks rather than stance stability, sycophancy, or truth-maintenance under adversarial social dynamics.

## Where GALILEO is stronger / cleaner (if true)

- If our claim is about *measuring* multi-turn degradation/drift in realistic dialogue and separating drift vs evidence-driven revision, that framing is orthogonal and (potentially) cleaner than conflating improvement with training objective.

## Where GALILEO is weaker / needs to improve

- We likely need a tighter bridge to training-time implications: how evaluation findings translate into mitigation/post-training (this paper is a concrete example of post-training leveraging multi-turn signals).

## Action items for GALILEO (experiments / method / writing)

- [ ] Related work positioning: cite as an example that multi-round interaction traces can be converted into a stronger learning signal than single-turn majority voting; contrast “improve via post-training on debate” vs “evaluate drift/instability under multi-turn pressure.”
- [ ] Consider an ablation idea: when evaluating instability, compare single-sample vs multi-sample aggregation (majority vote / debate-style aggregation) as *defenses* and quantify residual drift.

## Quotes / details to potentially cite

- “Multi-agent debate… provides an even richer signal than single-round majority voting.”
- “Preference learning over full reasoning traces… more effective than binary consensus rewards or SFT-based approaches.”
- Reported improvements: “+26.87% on MATH… +21.51% on MathQA… +27.6% on GSM8K… +16.3% on GPQA… +11.6% on CommonsenseQA.”
