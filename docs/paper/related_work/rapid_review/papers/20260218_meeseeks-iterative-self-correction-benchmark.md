# Meeseeks: A Feedback-Driven, Iterative Self-Correction Benchmark evaluating LLMs’ Instruction Following Capability

- Year: 2025
- Venue: arXiv
- Authors: Jiaming Wang, Yunke Zhao, Peng Ding, Jun Kuang, Yibin Shen, Zhe Tang, Yilin Jin, ZongYu Wang, Xiaoyu Li, Xuezhi Cao, Xunliang Cai
- URL: https://arxiv.org/abs/2504.21625
- BibTeX key (if we add it): Meeseeks2025
- Tags: multi-turn, instruction-following, iterative, evaluation, self-correction, feedback

## One-sentence takeaway

Meeseeks benchmarks how far LLMs can get on *multi-constraint instruction following* when given targeted, turn-by-turn feedback—and finds that even after up to 20 correction rounds, models still often fail to satisfy all constraints.

## What problem does it solve?

- Existing instruction-following benchmarks are often *single-shot*; they do not measure the “upper bound” achievable via iterative self-correction.
- Automatic evaluation of many fine-grained constraints is expensive/brittle; they propose a hybrid evaluation pipeline to improve accuracy and cost.

## What is the core method / protocol?

- A dataset of **700+** curated multi-constraint instruction-following instances, annotated with **32 capability tags**, in **Chinese + English**.
- An *iterative loop*: (1) model answers, (2) evaluator detects which constraints are violated and why, (3) feedback is provided, (4) model revises; repeated for multiple turns (reported up to **20**).
- Automatic evaluation uses **code-guided, rule-augmented LLM-based evaluation**:
  - Deterministic checks when possible (structure / regex-like / countable constraints).
  - LLM judging for constraints that require semantic interpretation, augmented with rules to reduce evaluator errors.

## What are the key metrics?

- Per-turn constraint satisfaction / task success rate across iterations (first-turn vs last-turn performance).
- Reported evaluator accuracy for their hybrid evaluation pipeline (compared to baselines).
- They also mention a “Meeseeks Score” and “Utility Rate” in the appendix (details not captured in this rapid skim).

## What are the main results?

- Their proposed evaluator improves evaluation accuracy **from 78.7% to 98.4%** (as stated in the paper introduction).
- Across **17** SOTA models, even with **20** rounds of feedback-driven self-correction, *no* model exceeds **91%** accuracy (as stated in the paper introduction).
- Large cross-model variance: some models benefit substantially from feedback; others plateau quickly.

## How is this similar to GALILEO?

- Protocol neighbor for **multi-turn robustness**: measures performance *over a trajectory* of interventions (feedback) rather than a single response.
- Emphasizes **fine-grained constraint checking** and the gap between first-try compliance vs best-achievable compliance after corrective signals.

## How is this different from GALILEO?

- Focus is *instruction/constraint compliance* with explicit feedback, not (primarily) belief revision / drift / susceptibility under social pressure.
- Uses an evaluator-centric pipeline (code + LLM judge) to label constraint violations, whereas GALILEO may care more about behavioral stability under different interaction regimes.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s goals include drift vs evidence-driven revision separation, Meeseeks is not directly targeted at that causal disentanglement.
- Meeseeks tasks are “does the output meet constraints?”; GALILEO can frame richer agent failure modes (e.g., persistence, recovery, intervention types).

## Where GALILEO is weaker / needs to improve

- Consider adopting Meeseeks-style **iterative feedback protocols** as a standardized “recovery curve” measure: not just whether a model fails, but *how quickly it can be guided back*.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an evaluation view that reports **turn-of-success / area-under-recovery-curve** when interventions are provided (feedback, hints, corrections).
- [ ] For any constraint-based tasks in GALILEO, consider a **hybrid evaluator** (code-first + LLM semantic checks) and report evaluator accuracy.
- [ ] Cite as evidence that “self-correction doesn’t fully solve it” even with many turns of feedback.

## Quotes / details to potentially cite

- “...even after 20 turns of iterative feedback-driven self-correction, nearly all models demonstrate suboptimal performance.” (arXiv abstract)
- “...raising accuracy from 78.7% to 98.4%.” (intro contribution summary)
- “...despite implementing 20 iterations... failed to achieve accuracy rates exceeding 91%.” (intro contribution summary)
