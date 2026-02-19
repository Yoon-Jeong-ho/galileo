# LLMs know their vulnerabilities: Uncover Safety Gaps through Natural Distribution Shifts

- Year: 2024 (arXiv), 2025 (ACL main)
- Venue: ACL 2025 Main Conference (arXiv:2410.10700)
- Authors: Qibing Ren, Hao Li, Dongrui Liu, Zhanxu Xie, Xiaoya Lu, Yu Qiao, Lei Sha, Junchi Yan, Lizhuang Ma, Jing Shao
- URL: https://arxiv.org/abs/2410.10700
- BibTeX key (if we add it): ren2024llms
- Tags: safety, multi-turn, distribution-shift, jailbreak, robustness

## One-sentence takeaway

Aligned LLMs can be driven into unsafe outputs via *natural* (in-distribution-looking) multi-turn prompts that are only semantically adjacent to toxic intents, using an actor-network-style expansion (ActorBreaker) rather than overt jailbreak templates.

## What problem does it solve?

- Identifies a safety failure mode not well-covered by standard refusal training: *natural distribution shifts* where benign, semantically related prompts can gradually elicit harmful content.
- Provides an automated way to generate diverse multi-turn attack trajectories that look like plausible user queries (vs. overt roleplay / hypothetical jailbreak patterns).

## What is the core method / protocol?

- **ActorBreaker**: given a toxic target query, construct an *actor network* inspired by Latour’s actor-network theory.
  - Six actor types (conceptual categories) connected to the harmful target (the paper emphasizes both **human** and **non-human** actors, e.g., people, books, media, movements).
  - Instantiate this network using the LLM’s own world knowledge (i.e., leverage pretraining priors to name related actors and relations).
- Sample actor nodes + relationships as **attack clues**, then generate a **multi-turn prompt chain** that starts seemingly harmless and incrementally increases proximity to the harmful target.
- Evaluation emphasizes:
  - diversity of generated attack prompts/paths,
  - success at eliciting unsafe content across aligned LLMs,
  - bypassing guard models (they mention Llama-Guard 2).
- Defense: construct a **multi-turn safety dataset** using ActorBreaker and fine-tune for improved robustness (with utility trade-offs).

## What are the key metrics?

- Attack **success rate** on HarmBench (and across aligned LLMs).
- Diversity / coverage of attack prompts (qualitative + likely quantitative diversity measures; paper claims improved diversity vs fixed-template multi-turn methods).
- Efficiency (number of turns / queries needed to succeed).
- Defense evaluation: robustness gains post fine-tuning, plus utility/safety trade-off.

## What are the main results?

- ActorBreaker outperforms prior single-turn and multi-turn attack baselines on HarmBench in **effectiveness**, **diversity**, and **efficiency**.
- Works even against stronger “reasoning” models (they explicitly mention GPT-o1) and can generate unsafe outputs.
- Generated prompts are “natural” enough to evade at least some guard-model detection (Llama-Guard 2 mentioned).
- Fine-tuning on their ActorBreaker-generated multi-turn safety data improves robustness, but can reduce utility.

## How is this similar to GALILEO?

- Shares the theme that **multi-turn interaction dynamics** matter: safety/robustness failures appear over turns, not just on a single query.
- Supports the argument that evaluations should consider **trajectory-based** or **turn-indexed** failure modes ("gradually lead" to failure).

## How is this different from GALILEO?

- Focuses on **unsafe-content elicitation** (jailbreak / refusal robustness) rather than agreement / belief / consistency phenomena per se.
- The “semantic neighbor” concept here is about *adjacent concepts in pretraining knowledge* (actor networks), not necessarily user pressure/sycophancy dynamics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets persuasion/sycophancy/stance drift, it can provide **cleaner causal decompositions** of *why* the model shifts beliefs/answers, beyond safety-policy gaps.

## Where GALILEO is weaker / needs to improve

- GALILEO should ensure it covers **natural-distribution-shift** style multi-turn prompts (benign-seeming trajectories) as a harder/realistic adversary class.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “natural semantic neighbor” condition: multi-turn conversations that approach a target claim/topic via related entities/events (no overt jailbreak framing) and measure turn-of-failure / drift.
- [ ] In related work, explicitly distinguish **malicious distribution shift** (template jailbreaks) vs **natural distribution shift** (semantic adjacency) as two robustness regimes.

## Quotes / details to potentially cite

- Abstract framing: susceptibility to “**natural distribution shifts** between attack prompts and original toxic prompts, where seemingly benign prompts, semantically related to harmful content, can bypass safety mechanisms.”
- ActorBreaker summary: identifies “actors related to toxic prompts within pre-training distribution to craft **multi-turn prompts that gradually lead** LLMs to reveal unsafe content,” grounded in Latour’s actor-network theory.
