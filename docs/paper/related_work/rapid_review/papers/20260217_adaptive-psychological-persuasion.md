# On the Adaptive Psychological Persuasion of Large Language Models

- Year: 2025
- Venue: arXiv
- Authors: Tianjie Ju; Yujia Chen; Hao Fei; Mong-Li Lee; Wynne Hsu; Pengzhou Cheng; Zongru Wu; Zhuosheng Zhang; Gongshen Liu
- URL: https://arxiv.org/abs/2506.06800
- BibTeX key (if we add it): ju2025adaptivepsychologicalpersuasion
- Tags: persuasion, psychology, strategies, dialogue, multi-turn, counterfactuals, DPO

## One-sentence takeaway

LLMs are mediocre at *autonomous* persuasion (tend to repeat weak tactics), but can become substantially more effective when guided (or trained via DPO) to *adaptively select* psychological persuasion strategies tailored to the context.

## What problem does it solve?

- Systematically evaluates the **dual capabilities** of LLMs to (i) persuade and (ii) resist persuasion in adversarial dialogues, with a focus on **psychological persuasion tactics**.
- Addresses that “generic” prompting yields repetitive strategies and low persuasion success.
- Proposes a way to make persuader LLMs **choose better tactics** rather than using a fixed, one-size-fits-all strategy.

## What is the core method / protocol?

- Setup: two roles, **persuader** and **listener** LLMs, interacting in adversarial dialogue.
- Data: uses the **CounterFact** dataset (factual triples with counterfactual alternatives) to define persuasion targets.
- Baseline finding: when asked to persuade without explicit strategy, models tend to use repetitive tactics → low success.
- Strategy prompting: defines **11 psychological persuasion strategies** (examples explicitly mentioned: *Fluency Effect*, *Repetition Effect*, *Scarcity Effect*) and prompts the persuader to use one.
- Adaptation via post-training: **Direct Preference Optimization (DPO)** fine-tuning to select strategies adaptively:
  - sample strategy-specific responses,
  - treat the resulting persuasion outcomes as **preference pairs**,
  - fine-tune to prefer responses/strategies that succeed.

## What are the key metrics?

- Primary: **persuasion success rate** (whether the listener adopts the counterfactual belief/object).
- Also framed as measuring both:
  - persuader’s ability to convince,
  - listener’s **epistemic resistance**.
- Domain breakdown: partitions evaluation set into **four semantic domains** (via GPT-4o) to test strategy-context dependence.

## What are the main results?

- Unguided persuader LLMs often employ repetitive/ineffective tactics → low persuasion success.
- Explicitly prompting a specific strategy can materially improve success, but effects are model- and context-dependent:
  - Example reported: **Scarcity Effect** improves success for LLaMA-3.1-8B-Instruct by **~15%** vs baseline prompting.
  - Example reported: **Fluency Effect** is particularly effective for GPT-4o scenarios.
- No universal best strategy across domains; effectiveness varies sharply by **semantic domain**.
- DPO adaptation: with **~3,000 training examples**, fine-tuned open models outperform originals across most strategies and show more diverse strategy selection, while claiming to maintain general capabilities.

## How is this similar to GALILEO?

- Both are about **multi-turn interaction dynamics** under *social/psychological pressure* (here: persuasion tactics; GALILEO: pressure-driven drift / instability).
- Reinforces the idea that surface-level “alignment” can be bypassed by well-chosen social/psychological operators.

## How is this different from GALILEO?

- Focus is **persuader-side optimization** (how to persuade better) rather than evaluating *robustness to pressure* with drift-vs-revision controls.
- Task is framed around counterfactual knowledge triples (CounterFact) and success/failure of belief adoption, not time-to-failure / recovery metrics.
- Does not foreground **trajectory metrics** (e.g., ToF/PWC/survival curves) or explicit **recovery-after-flip** objectives.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes paired conditions separating **evidence-based revision** vs **pressure-only drift**, that’s a cleaner causal story than persuasion-only success rates.
- GALILEO’s emphasis on **multi-turn failure dynamics** (when failure happens, recovery, oscillation) is richer than a single success-rate endpoint.

## Where GALILEO is weaker / needs to improve

- This paper suggests there may be meaningful structure in *which persuasion operator is used* (fluency, scarcity, repetition, etc.). If GALILEO’s pressure operators are too coarse, we may miss operator-specific vulnerabilities.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a small operator taxonomy for “pressure moves” inspired by their 11 strategies (at least 3–5), and report robustness by operator.
- [ ] Consider an adversary baseline that is *adaptive* (selects the best pressure strategy per instance) to stress-test robustness claims.
- [ ] In related work, cite as evidence that (i) LLM persuasion is nontrivial without prompting, but (ii) strategy selection + lightweight post-training can significantly strengthen persuasion.

## Quotes / details to potentially cite

- Abstract: proposes an “adaptive framework based on direct preference optimization” to select optimal strategies; shows improvements while “maintaining general capabilities.”
- Intro: notes persuader LLMs’ “repetitive and ineffective persuasion tactics” under baseline prompting.
- Intro (example): “Scarcity Effect boosts success rate in LLaMA-3.1-8B-Instruct scenarios by 15% compared to baseline prompts.”
