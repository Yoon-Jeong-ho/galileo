# Beyond Single-Turn: A Survey on Multi-Turn Interactions with Large Language Models

- Year: 2025
- Venue: arXiv (survey)
- Authors: Yubo Li, Xiaobin Shen, Xinyu Yao, Xueying Ding, Yidi Miao, Ramayya Krishnan, Rema Padman
- URL: https://arxiv.org/abs/2504.04717
- BibTeX key (if we add it): li2025beyond
- Tags: survey, multi-turn, evaluation, benchmarks, robustness, context

## One-sentence takeaway

A task-oriented survey of **multi-turn LLM interactions** that (i) systematizes benchmark families and evaluation criteria and (ii) organizes improvement methods (model-centric, external integration, agent-based), with an explicit “open challenges” taxonomy.

## What problem does it solve?

- Provides a consolidated map of the fast-growing literature on **multi-turn interaction** beyond single-turn instruction following.
- Clarifies that real deployments require maintaining **context, coherence, responsiveness**, and robustness across turns; argues that single-turn benchmarks miss compounding-error dynamics.

## What is the core method / protocol?

- Survey / taxonomy (no new primary experimental protocol).
- Organizes multi-turn work primarily by **tasks** rather than “capabilities”:
  - **Instruction-following** (general, math, coding)
  - **Conversational engagement** (roleplay, healthcare, education, jailbreak)
- Catalogs improvement methods in three buckets:
  - **Model-centric**: in-context learning, SFT, RL, architecture changes
  - **External integration**: memory, retrieval, knowledge graphs
  - **Agent-based**: single-agent and multi-agent frameworks

## What are the key metrics?

- As a survey, it primarily *summarizes* metrics used across benchmarks rather than proposing a new one.
- Examples highlighted in benchmark summaries include:
  - Multi-turn instruction-following criteria such as **helpfulness, relevance, accuracy, depth**, constraint satisfaction, and session stability.
  - “LLM-as-a-judge” agreement rates with humans (e.g., MT-Bench-style judging).
  - Multi-turn degradation patterns (error compounding, dependence on temporal distance to relevant context).
  - The survey explicitly calls out the need for **long-term effectiveness** evaluation (i.e., metrics that reflect sustained performance over turns).

## What are the main results?

- Not an empirical paper; main contributions are organizational:
  - A fairly comprehensive benchmark table for multi-turn instruction following and evaluation designs.
  - A view that multi-turn evaluation should reflect real-world **dynamic user intent** and **compounding failure**.
  - A structured set of open challenges across: context management, complex reasoning, adaptation/learning, evaluation methodology, and ethics/safety.

## How is this similar to GALILEO?

- Motivates why GALILEO-style work matters: **multi-turn** settings expose failure modes that single-turn tests miss (drift, compounding errors, susceptibility to follow-ups).
- Provides a convenient umbrella citation for:
  - “multi-turn robustness needs dedicated metrics and benchmarks”
  - “turn-by-turn evaluation and long-horizon effects”
- Mentions (via cited benchmark summaries) metrics and ideas GALILEO already leans on/adjacent to, e.g., **Position-Weighted Consistency (PWC)** and “session stability” style measures.

## How is this different from GALILEO?

- Survey vs. a new protocol/metric paper: it doesn’t propose a **pressure-vs-evidence control** design, **survival/time-to-failure** measurement, or **recovery-after-flip trajectories**.
- Broader scope; includes many task domains (education/healthcare/jailbreak) not necessarily aligned with GALILEO’s core *social pressure / belief drift* emphasis.

## Where GALILEO is stronger / cleaner (if true)

- Can make a sharper causal contrast between:
  - **evidence-driven revision** vs.
  - **pressure-driven drift**
- Can report trajectory-aware robustness (time-to-event / recovery) rather than only “quality averaged over turns”.

## Where GALILEO is weaker / needs to improve

- Could benefit from adopting some of the survey’s **task-oriented framing** for the paper narrative (so readers understand what “multi-turn robustness under pressure” is a *task family* of).
- Might need a clearer placement among general multi-turn benchmarks (MT-Bench variants, MT-Eval, etc.) to pre-empt “why another benchmark?” questions.

## Action items for GALILEO (experiments / method / writing)

- [ ] Use this survey as a high-level related-work anchor for the claim: “multi-turn interaction is a distinct evaluation regime; single-turn results don’t transfer reliably.”
- [ ] Consider mirroring its **task-oriented taxonomy** when positioning GALILEO (e.g., define GALILEO’s target as a particular class of multi-turn interaction: *social-pressure / belief-stability tasks*).
- [ ] Add 1–2 sentences in related work that connect GALILEO’s metrics (e.g., time-to-failure / recovery) to the survey’s “long-term effectiveness” evaluation gap.

## Quotes / details to potentially cite

- Abstract-level framing: real-world multi-turn interactions require maintaining “context, coherence, fairness, and responsiveness” over prolonged dialogue.
- Scope statement: the survey focuses on text-only multi-turn interactions and explicitly excludes multimodal to stay focused.
- The survey’s open-challenges section includes “evaluation on long-term effectiveness” as a stated need (useful to motivate survival/time-to-failure + recovery metrics).
