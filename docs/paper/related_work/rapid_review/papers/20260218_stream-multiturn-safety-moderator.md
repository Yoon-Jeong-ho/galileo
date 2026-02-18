# SafeTy Reasoning Elicitation Alignment for Multi-Turn Dialogues (STREAM)

- Year: 2025
- Venue: arXiv
- Authors: Martin Kuo, Jianyi Zhang, Aolin Ding, Louis DiValentin, Amin Hass, Benjamin F Morris, Isaac Jacobson, Randolph Linderman, James Kiessling, Nicolas Ramos, Bhavna Gopal, Maziyar Baran Pouyan, Changwei Liu, Hai Li, Yiran Chen
- URL: https://arxiv.org/abs/2506.00668
- BibTeX key (if we add it): kuo2025stream
- Tags: multi-turn, safety, jailbreak, defense, reasoning-moderator

## One-sentence takeaway

A plug-and-play “safety reasoning moderator” fine-tuned on a human-annotated multi-turn jailbreak dataset can significantly reduce multi-turn attack success rates while preserving general capability.

## What problem does it solve?

- Multi-turn jailbreak / harmful-intent attacks where an adversary gradually steers the model over several turns, hiding malicious intent until late in the dialogue.
- Practicality gap: per-model safety fine-tuning is expensive and does not transfer; existing moderation systems struggle with multi-turn intent.

## What is the core method / protocol?

- Build a **Safety Reasoning Multi-turn Dialogue** dataset (2,177 dialogues):
  - Human annotators label each turn for attack intent.
  - If malicious, assign one/more of **37 malicious categories** + a **severity (0–10)**.
- Elicit **“metacognitive” chain-of-thought safety reasoning** behind each judgment using a large reasoning model.
- **SFT a separate moderator model** (“safety reasoning moderator”) on this annotated+reasoning data.
- **Deployment:** insert the moderator between user and target LLM.
  - If malicious multi-turn intent is detected, moderator **appends a warning prompt** to the user query to alert the downstream LLM (rather than hard-blocking).

## What are the key metrics?

- **Attack Success Rate (ASR)** under prevalent multi-turn attack strategies.
- Capability retention: reported via standard benchmarks like **MMLU** and **GSM8K** (and “comparable capability” claims).

## What are the main results?

- Reported ASR reduction (headline): **51.2%** reduction vs existing defenses (aggregate claim on arXiv abstract).
- Per-model examples (from HTML version):
  - GPT-4.1: average ASR reduction **48.7%** vs baseline defenses.
  - o4-mini: average ASR reduction **26.3%**.
  - LLaMA-3.1-Nemotron-Nano-8B: average ASR reduction **27.1%**.
- Claims capability is **comparable** to baselines (no major regression on MMLU/GSM8K).

## How is this similar to GALILEO?

- Targets **multi-turn robustness under pressure/adversarial steering**, focusing on latent intent emerging across turns.
- Uses an explicit **reasoning/interpretation layer** (moderator) rather than only single-turn filters.
- Emphasizes **plug-and-play** defenses that can generalize across different base models.

## How is this different from GALILEO?

- STREAM is explicitly a **safety/jailbreak defense** framing (attack success reduction), whereas GALILEO’s core framing (as per rapid-review goals) includes broader multi-turn stability phenomena (drift, sycophancy/persuasion, belief revision control).
- STREAM’s key intervention is **an external moderator that appends warnings**, not a change to the target model’s internal update/stability behavior.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides **task-general multi-turn stability guarantees/measurements** (beyond jailbreak domains), it can claim broader relevance than an attack-specific ASR metric.
- If GALILEO avoids reliance on chain-of-thought style rationales, it may be cleaner under “reasoning visibility” constraints.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a strong **multi-turn adversarial** evaluation suite, STREAM suggests a concrete benchmark axis: multi-turn ASR vs established attack strategies.
- If GALILEO doesn’t have a modular “interpose a model” story, STREAM’s deployment simplicity is compelling.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add/expand an evaluation slice: multi-turn adversarial steering where malicious goal is revealed late; track a STREAM-like **ASR** metric alongside GALILEO metrics.
- [ ] Consider a GALILEO ablation: add an “intent-warning” preface (moderator-style) to see how much robustness comes from simple alerting vs deeper method.
- [ ] Related-work positioning: cite STREAM as evidence that **multi-turn intent reasoning** needs special handling and that “moderation” must be multi-turn aware.

## Quotes / details to potentially cite

- Abstract (core claim): “Experimental results demonstrate that our method significantly outperforms existing defense techniques, reducing the Attack Success Rate (ASR) by **51.2%**, all while maintaining comparable LLM capability.”
- Dataset detail: **2,177** multi-turn dialogues; **37** malicious categories; severity **0–10**.
- Deployment description: moderator “**appends a warning prompt** to the original query” when malicious intent is detected.
