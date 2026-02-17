# Foot-In-The-Door: A Multi-turn Jailbreak for LLMs

- Year: 2025
- Venue: arXiv
- Authors: ZIxuan Weng et al. (authors not listed in fetched abstract page; verify later if needed)
- URL: https://arxiv.org/abs/2502.19820
- BibTeX key (if we add it): weng2025fitd
- Tags: multi-turn, jailbreak, foot-in-the-door, self-corruption, alignment

## One-sentence takeaway

FITD is a multi-turn jailbreak strategy that starts with small “benign-ish” commitments and progressively escalates to harmful requests, achieving very high attack success rates across multiple models.

## What problem does it solve?

- Shows that single-turn safety evaluations miss an important failure mode: *multi-turn* conversational dynamics can gradually weaken refusals.
- Provides a concrete attack protocol (and benchmarks/results) for studying “self-corruption” / alignment drift over dialogue.

## What is the core method / protocol?

- **Foot-in-the-door (FITD) multi-turn prompting**:
  - Start with minor, low-stakes prompts to elicit cooperation.
  - Use **intermediate “bridge” prompts** that incrementally increase malicious intent.
  - Leverage the model’s own prior responses (“aligns the model's response by itself”) to continue the trajectory toward disallowed content.
- Emphasis is on **progressive escalation** rather than an immediate overtly malicious prompt.

## What are the key metrics?

- **Attack Success Rate (ASR)** on two jailbreak benchmarks (not named in the abstract).
- Aggregate ASR across **seven widely used models**.

## What are the main results?

- Reports **~94% average ASR** across seven models on two benchmarks, outperforming prior jailbreak methods (per abstract).
- Includes analysis framing the phenomenon as **LLM self-corruption** in multi-turn settings.

## How is this similar to GALILEO?

- If GALILEO involves multi-step interactions (agentic workflows, iterative refinement, conversational protocols), FITD is directly relevant: it demonstrates **stateful, multi-turn context** can be exploited to bypass safeguards.
- Highlights that safety/alignment should be evaluated over **trajectories** (turn-by-turn), not only pointwise prompts.

## How is this different from GALILEO?

- FITD is an **offensive** jailbreak method; GALILEO (presumably) is not a jailbreak technique and may focus on robust behavior, evaluation, or safer protocols.
- FITD’s mechanism centers on **escalation + commitment** dynamics, rather than model capability improvements.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO enforces explicit policy constraints, memory sanitation, or turn-level safety checks, it may be more robust than “naive chat” settings targeted by FITD.

## Where GALILEO is weaker / needs to improve

- Any GALILEO component that:
  - trusts prior model outputs as “safe to build on”,
  - chains multiple turns without re-evaluating safety at each step,
  - or uses summarization / self-reflection that can preserve unsafe intent,
  may inherit FITD-style vulnerabilities.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a **multi-turn red-team eval** where malicious intent is escalated over 3–10 turns (include “bridge prompt” patterns).
- [ ] Track **refusal stability** over dialogue (e.g., first refusal turn vs later turns; quantify drift).
- [ ] If GALILEO uses any “self-critique/self-rewrite” loops, test whether the loop can be steered into unsafe continuations after benign starts.
- [ ] In related work, cite FITD as evidence that **multi-turn interaction is a distinct jailbreak surface**.

## Quotes / details to potentially cite

- “Inspired by psychological foot-in-the-door principles… [a] novel multi-turn jailbreak method…” (abstract)
- “achieves an average attack success rate of 94% across seven widely used models…” (abstract)
- “in-depth analysis of LLM self-corruption… risks inherent in multi-turn interactions.” (abstract)
