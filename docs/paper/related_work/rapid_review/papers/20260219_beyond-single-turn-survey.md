# Beyond Single-Turn: A Survey on Multi-Turn Interactions with Large Language Models

- Year: 2025
- Venue: arXiv (survey)
- Authors: Yubo Li; Xiaobin Shen; Xinyu Yao; Xueying Ding; Yidi Miao; Ramayya Krishnan; Rema Padman
- URL: https://arxiv.org/html/2504.04717v1
- BibTeX key (if we add it): li2025beyond
- Tags: survey, multi-turn, dialogue, evaluation, robustness

## One-sentence takeaway

A broad, task-oriented survey of multi-turn LLM interaction benchmarks and improvement methods, with a useful taxonomy (instruction-following vs conversational engagement) and an explicit multi-turn challenge list (context retention, drift, evaluation, and safety).

## What problem does it solve?

- Single-turn evaluation/training dominates, but real deployments are multi-turn; the paper tries to organize the fragmented multi-turn landscape.
- Provides an overview of (i) multi-turn tasks and benchmarks/datasets, (ii) evaluation criteria and judge choices, and (iii) methods to improve multi-turn performance.

## What is the core method / protocol?

- Survey / taxonomy paper.
- Organizes multi-turn work along two task families:
  - Instruction following (general, math, coding).
  - Conversational engagement (roleplay, healthcare, education, adversarial/jailbreak).
- Separately categorizes improvement approaches:
  - Model-centric: in-context learning; supervised fine-tuning; reinforcement learning (mentions multi-turn DPO variants, credit assignment themes); architectural changes.
  - External integration: memory-augmented; retrieval-augmented; knowledge-graph integration.
  - Agent-based approaches: single-agent; multi-agent (role-based collaboration; debate; dynamic composition).
- Includes an “open challenges” taxonomy: context understanding/management; complex reasoning across turns (error propagation, compounding); adaptation/learning; evaluation methodology; ethics/safety.

## What are the key metrics?

- Not a single metric contribution; instead catalogs evaluation criteria used in prior work.
- Recurrent dimensions highlighted (useful for GALILEO’s framing):
  - Context retention + coherence across turns.
  - Consistency / stability (avoid drift).
  - Responsiveness / helpfulness.
  - Fairness / bias.
  - Safety under multi-turn adversarial pressure (e.g., jailbreak).
  - Human vs LLM judge considerations and agreement checks.

## What are the main results?

- Main “result” is a structured map of datasets/benchmarks and method families.
- Practical artifact: curated resource list / repo: https://github.com/yubol-cmu/Awesome-Multi-Turn-LLMs

## How is this similar to GALILEO?

- Same high-level concern: robustness over multiple rounds (drift, compounding errors, maintaining objectives under interaction pressure).
- Explicitly treats adversarial/jailbreak as multi-turn phenomena, aligning with “pressure over rounds” framing.

## How is this different from GALILEO?

- Broad survey rather than a targeted protocol for “multi-turn robustness under pressure” or sycophancy/persuasion control.
- Focuses on taxonomy and coverage; not primarily about causal interventions or controlled experimental designs for drift/sycophancy.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a concrete, controlled benchmark/protocol for multi-turn robustness (e.g., explicit pressure tactics, standardized scoring, ablations), it can be positioned as *more operational* than survey-level categorization.
- GALILEO can contribute sharper definitions/metrics for “stability vs warranted belief revision” beyond generic “consistency”.

## Where GALILEO is weaker / needs to improve

- The survey’s breadth can expose missing coverage in GALILEO’s related work, especially:
  - Domain-specific multi-turn applications (healthcare/education) if GALILEO claims generality.
  - External integration baselines (memory/RAG/knowledge graph) as drift mitigations.
  - Agent-based multi-turn setups (debate, role-based collaboration) as both threat models and mitigation strategies.

## Action items for GALILEO (experiments / method / writing)

- [ ] Use this as an “umbrella” citation motivating why single-turn is insufficient; add 1–2 sentences in related work introducing their two-task taxonomy.
- [ ] Cross-check their benchmark tables for any *multi-turn evaluation datasets* we should explicitly include as baselines or comparisons.
- [ ] In GALILEO writing, map GALILEO’s threat model(s) into their open-challenges taxonomy (context management; error propagation; adaptation; evaluation; safety).
- [ ] Mine the linked “Awesome-Multi-Turn-LLMs” list for additional candidates directly relevant to (a) sycophancy under rebuttal, (b) persuasion dynamics, (c) drift/equilibria.

## Quotes / details to potentially cite

- Abstract-level framing: “real-world applications demand sophisticated multi-turn interactions” and multi-turn brings challenges around “maintaining context, coherence, fairness, and responsiveness over prolonged dialogues.”
- The paper’s explicit scope note: focuses on text-based multi-turn, excludes multimodal to keep analysis focused (useful if GALILEO is also text-only).
