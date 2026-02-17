# Foot-In-The-Door: A Multi-turn Jailbreak for LLMs

- Year: 2025
- Venue: arXiv
- Authors: Zixuan Weng et al.
- URL: https://arxiv.org/abs/2502.19820
- BibTeX key (if we add it): foot_in_the_door_2025
- Tags: multi-turn, jailbreak, gradual-escalation, safety

## One-sentence takeaway

FITD is a psychologically inspired multi-turn jailbreak that escalates from benign prompts to harmful requests and reports very high attack success rates across multiple popular LLMs, highlighting multi-turn “self-corruption” risks.

## What problem does it solve?

- Shows that safety-aligned LLMs can be systematically coerced into disallowed outputs via *multi-turn* interaction patterns that build incremental commitment.
- Argues single-turn jailbreak defenses/benchmarks can miss an important failure mode: gradual intent escalation.

## What is the core method / protocol?

- **Foot-in-the-door (FITD) attack**:
  - Start with a benign / low-stakes query related to the eventual harmful topic.
  - Use **bridge prompts** that *progressively* increase malicious intent.
  - Uses the model’s own prior responses to “align the model’s response by itself” (author phrasing) as the conversation escalates.
- Evaluated as an automated multi-turn attack on two jailbreak benchmarks (not detailed on the abstract page).

## What are the key metrics?

- **Attack Success Rate (ASR)** of producing “harmful disallowed outputs” under the jailbreak protocol.

## What are the main results?

- Reports **~94% average ASR** across **7 widely used models** on **two jailbreak benchmarks**, outperforming prior methods (per abstract).

## How is this similar to GALILEO?

- Both are concerned with **multi-turn dynamics** and how model behavior can drift across turns under interaction pressure.
- FITD’s “gradual escalation” is an instance of **sequential vulnerability accumulation**, which is conceptually close to multi-turn robustness/stability framing.

## How is this different from GALILEO?

- FITD is an **attack method** (jailbreaking) rather than a general-purpose framework for measuring/helping *robustness under conversational drift*.
- Primary outcome is **safety policy violation** (harmful content) vs. GALILEO’s broader interest in behavioral/epistemic stability under multi-turn perturbations.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can position itself as more **general** (not only safety-jailbreak), with clearer experimental controls for *types* of drift/pressure (if we include them).

## Where GALILEO is weaker / needs to improve

- If GALILEO doesn’t include **gradual escalation protocols**, it may miss an important class of multi-turn failures that are easy for humans to execute.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “**gradual escalation / commitment**” multi-turn stress test variant (benign → bridge → target prompt) and measure degradation curves.
- [ ] In related work, cite FITD as evidence that **multi-turn interaction structure** (not just content) drives safety/robustness failure.
- [ ] If we already report survival/time-to-failure metrics, note FITD-style attacks as a concrete mechanism that can induce earlier failure.

## Quotes / details to potentially cite

- “Inspired by psychological foot-in-the-door principles, we introduce FITD, a novel multi-turn jailbreak method…”
- “Extensive experimental results on two jailbreak benchmarks demonstrate that FITD achieves an average attack success rate of 94% across seven widely used models…”
- “we provide an in-depth analysis of LLM self-corruption… emphasizing the risks inherent in multi-turn interactions.”
