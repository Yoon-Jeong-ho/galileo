# Evaluating & Reducing Deceptive Dialogue From Language Models with Multi-turn RL

- Year: 2025
- Venue: arXiv
- Authors: Marwa Abdulhai; Ryan Cheng; Aryansh Shrivastava; Natasha Jaques; Yarin Gal; Sergey Levine
- URL: https://arxiv.org/abs/2510.14318
- BibTeX key (if we add it): Abdulhai2025DeceptiveDialogueMultiTurnRL
- Tags: multi-turn, deception, robustness, RL, trajectory-metrics

## One-sentence takeaway

Proposes a *listener-effect* deception metric (“belief misalignment”) for multi-turn dialogue, finds it matches human judgments better than prior deception metrics, and shows multi-turn RL fine-tuning can substantially reduce deceptive dialogue behavior.

## What problem does it solve?

- Measuring deception in LLM dialogue in a way that reflects *impact on the listener* (not just whether an utterance is false/misleading in isolation).
- Evaluating deception as an emergent multi-turn behavior (single-turn metrics miss interaction dynamics).
- Mitigating deception with training that directly targets the multi-turn interaction setting.

## What is the core method / protocol?

- Simulate multi-turn dialogues between two prompted agents:
  - **DD (deceiver/speaker)**: an LLM with a task objective; may be explicitly prompted to deceive or not.
  - **LL (listener)**: a naive LLM not prompted to suspect deception.
- Use a third **LLM-as-judge** conditioned on ground-truth “world facts” to score each conversation/turn.
- Evaluate across **four dialogue scenarios** (paper example: buyer–seller negotiation; the paper states four distinct scenarios/datasets).
- Mitigation: a **multi-turn reinforcement learning** fine-tuning pipeline using deception-specific reward (targeting reduced belief misalignment / deception).

## What are the key metrics?

From the paper’s methodology section:

- **Belief misalignment (proposed):** at each turn, query judge for listener’s inferred beliefs; score deviation from the true world state. Intended to capture deception via *belief change*.
- **Deception Count:** fraction of deceiver turns judge labels as deceptive.
- **Deception Rating:** average Likert (1–5) deceptiveness rating per deceiver turn.
- Plus “established deception detection metrics” (paper mentions five established metrics total; full definitions/prompts are in appendix).

## What are the main results?

- The proposed **belief misalignment** metric correlates more closely with **human deception judgments** than the other tested metrics.
- Across **eight** evaluated LLMs, “benign” prompts still yield deceptive behavior in about **26%** of dialogue turns (per their measurement).
- When explicitly prompted to deceive, models can increase deceptiveness by up to **31%** relative to baseline.
- **RLHF-trained** models (as a class) still exhibit high deception rates (reported ~**43%** on average).
- Their **multi-turn RL** fine-tuning reduces deceptive behaviors by **77.6%** compared to other instruction-tuned models (as reported).

## How is this similar to GALILEO?

- Treats undesirable behavior as a **trajectory property** in multi-turn interaction (not a single response failure).
- Emphasizes evaluation designs where **pressure/strategy emerges over turns**, which aligns with GALILEO’s multi-turn robustness framing.
- Reinforces the idea that post-training (incl. RLHF) can leave substantial “socially risky” behaviors in the wild.

## How is this different from GALILEO?

- Target behavior is **deception** (belief manipulation), not specifically **social-pressure sycophancy / agreement drift**.
- Uses a **listener-belief modeling** approach (LLM-judge infers listener beliefs) rather than paired neutral-vs-pressure controls for the *same question*.
- Focuses on mitigation via **multi-turn RL**, not primarily on disentangling *evidence-driven revision vs pressure-driven drift* or *recovery-after-flip*.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes explicit **neutral vs pressure** counterfactuals and **recovery** phases, it can more cleanly attribute changes to *pressure operators* and quantify *return-to-truth* after perturbation.

## Where GALILEO is weaker / needs to improve

- This paper’s **belief misalignment** framing suggests we should consider listener-impact metrics (or at least cite this as precedent for multi-turn “effect on beliefs” evaluation).
- They provide a concrete mitigation baseline (multi-turn RL) we may want to acknowledge/compare against.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short “related metrics” paragraph contrasting **flip-based** metrics (ToF/NoF, survival) with **listener-effect** metrics (belief misalignment), and argue why GALILEO’s controls/recovery complement theirs.
- [ ] Consider a small pilot where a judge infers a “naive listener’s belief” over turns to see whether our pressure protocols measurably shift beliefs (even when answers remain verbally hedged).
- [ ] In mitigation baselines, consider positioning **multi-turn RL** as a neighbor approach (even if we do not implement it).

## Quotes / details to potentially cite

- “Given that deception in dialogue is a behavior that develops over an interaction history, its effective evaluation and mitigation necessitates moving beyond single-utterance analyses.” (Intro/Abstract)
- Reported toplines (Abstract): ~26% deceptive turns under benign prompts; up to +31% deceptiveness when prompted; RLHF models ~43% deception on average; multi-turn RL reduces deceptiveness by 77.6%.
