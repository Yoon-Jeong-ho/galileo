# How Social is It? A Benchmark for LLMs' Capabilities in Multi-user Multi-turn Social Agent Tasks

- Year: 2025
- Venue: arXiv
- Authors: Yusen Wu, Junwu Xiong, Xiaotie Deng
- URL: https://arxiv.org/abs/2505.04628
- BibTeX key (if we add it): hsii2025
- Tags: multi-user, multi-turn, social-agents, benchmark, target-selection, target-switching, cot

## One-sentence takeaway

HSII proposes a staged benchmark for multi-user, multi-turn “social agent” behavior (who to talk to, how to switch targets, and how to sustain dialogue), plus a cost-aware metric (COT-complexity) to evaluate whether chain-of-thought improves performance efficiently.

## What problem does it solve?

- Existing LLM benchmarks mostly test dyadic (single-user) dialogue or single-turn skills, and don’t systematically measure competencies needed for realistic multi-user social settings (selecting interlocutors, switching targets, maintaining stability).
- The authors want an evaluation protocol that decomposes multi-user social interaction into measurable subskills and yields a single aggregate score.

## What is the core method / protocol?

- Introduces a sociologically motivated task-leveling / decomposition for “social agent tasks”.
- Builds **HSII** benchmark with **four stages** (as described in the paper’s abstract/intro):
  1) **Format parsing** (follow required response format)
  2) **Target selection** (choose who to address among multiple participants)
  3) **Target switching conversation** (produce appropriate transitional utterances while switching interlocutors)
  4) **Stable conversation** (maintain coherent multi-turn conversation after switching)
- Dataset construction: starts from a **news dataset**, then algorithmic clustering + “detoxification”, then LLM-assisted summarization/refinement (GPT-4 mentioned), then human review/edits to create multi-participant scenarios with conflicts.
- Studies impact of **chain-of-thought (CoT)** prompting; adds **COT-complexity** to quantify cost/efficiency of using CoT for a target accuracy threshold.

## What are the key metrics?

- **HSII score** (aggregate across the four stages; details likely in the full paper).
- **COT-complexity**: a statistical metric intended to measure the “efficiency” of a model under CoT prompting (minimum reasoning cycles / reflection steps needed to meet an accuracy threshold, per the intro).

## What are the main results?

- The paper reports that HSII is suitable for evaluating “social skills” in LLMs, and that CoT can affect performance but has compute cost; COT-complexity is proposed to evaluate that trade-off.
- (Within this rapid review window, I relied on abstract + intro; concrete leaderboard numbers/models require reading the evaluation section/tables.)

## How is this similar to GALILEO?

- Both care about **multi-turn** capability evaluation beyond single-turn accuracy.
- The staged decomposition (parsing → selection → switching → stability) is conceptually similar to breaking an interaction into **failure modes** and measuring robustness over a trajectory.

## How is this different from GALILEO?

- HSII is focused on **multi-user social interaction mechanics** (who to talk to, switching targets, conversation stability) rather than (presumably) GALILEO’s primary phenomena.
- HSII is scenario-driven from **news-derived social situations**, whereas GALILEO may use different task generators / perturbations / adversarial setups.
- Introduces an explicit “cost-aware” metric (COT-complexity) tied to CoT use.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses more controlled perturbations and clearer causal knobs, it may yield cleaner attribution of *why* a model fails (vs. the broader realism of social scenarios).

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks **multi-user** interactions, HSII highlights missing evaluation dimensions: target selection, target switching, and post-switch stability.
- If GALILEO reports accuracy without cost normalization, HSII’s **COT-complexity** suggests adding compute-aware comparisons.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a “multi-user / multi-target” extension track (even a toy version) to separate: target selection errors vs. post-switch coherence failures.
- [ ] Add a cost-aware analysis for any “reasoning prompt” intervention (e.g., report quality vs. token/steps; HSII’s COT-complexity is one possible framing).
- [ ] Cite HSII as related work for multi-user multi-turn benchmarks and staged evaluation protocols.

## Quotes / details to potentially cite

- “HSII comprises four stages: format parsing, target selection, target switching conversation, and stable conversation…” (abstract)
- Dataset derived step-by-step from news; includes clustering + detoxification + LLM refinement + human curation to create multi-participant conflict scenarios (intro summary)
- “We further introduce a new statistical metric, COT-complexity, to quantify the efficiency of certain LLMs with COTs…” (abstract)
