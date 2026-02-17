# Evaluating LLM-based Agents for Multi-Turn Conversations: A Survey

- Year: 2025
- Venue: arXiv
- Authors: Shengyue Guan; Jindong Wang; Jiang Bian; Bin Zhu; Jian-guang Lou; Haoyi Xiong
- URL: https://arxiv.org/abs/2503.22458
- BibTeX key (if we add it): Guan2025EvaluatingLLMAgentsMultiTurnSurvey
- Tags: survey, multi-turn, evaluation, agents

## One-sentence takeaway

A survey that organizes **what** to evaluate in multi-turn LLM agents (capabilities/dimensions) and **how** to evaluate them (human, automatic, hybrid, self-judge), using a PRISMA-style literature review.

## What problem does it solve?

- Multi-turn conversational agents are evaluated with a fragmented set of tasks, metrics, and human protocols.
- The paper aims to systematize evaluation by offering taxonomies that cover both *agent components/dimensions* and *evaluation methodologies*.

## What is the core method / protocol?

- PRISMA-inspired systematic survey of ~250 sources (per abstract).
- Two interrelated taxonomy systems:
  - **What to evaluate**: key components/dimensions for multi-turn conversational agents.
  - **How to evaluate**: methodology categories (human annotation, automated metrics, hybrids, LLM self-judging).

## What are the key metrics?

As a survey, it does not introduce a single new metric; it catalogs common choices, including:

- Traditional NLG/LU metrics used (sometimes questionably) for dialogue: BLEU/ROUGE (explicitly mentioned in abstract).
- Multi-turn-specific dimensions (evaluation targets):
  - task completion
  - response quality
  - user experience
  - memory + context retention
  - planning
  - tool integration

## What are the main results?

- Consolidates evaluation dimensions for multi-turn LLM agents into a holistic checklist (what-to-evaluate taxonomy).
- Consolidates evaluation methodology families (how-to-evaluate taxonomy), explicitly including LLM-as-a-judge / self-judging approaches.

## How is this similar to GALILEO?

- Shares the emphasis that **multi-turn** settings require different evaluation thinking than single-turn.
- Provides a “map of the space” that can justify GALILEO’s focus on particular dimensions (robustness under social pressure; longitudinal dynamics; recovery).

## How is this different from GALILEO?

- This work is a **survey/taxonomy**, not a new benchmark for social pressure / sycophancy trajectories.
- It does not (from the abstract-level view) foreground **time-to-failure / survival-style** metrics, pressure operators, or drift-vs-revision controls.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can contribute an evaluation protocol with clearer causal structure for **pressure-induced drift** vs **evidence-driven revision**, and include **time-to-event** and **recovery** measures.

## Where GALILEO is weaker / needs to improve

- GALILEO should ensure broad coverage/justification of evaluation dimensions (user experience, tool use, memory) so readers don’t view it as overly narrow.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short “positioning vs agent-evaluation surveys” paragraph: GALILEO targets a specific axis (social-pressure robustness over turns), complementary to broad taxonomies.
- [ ] Consider adopting their *dimension taxonomy language* to label GALILEO’s measured components (e.g., memory/context retention vs belief stability).

## Quotes / details to potentially cite

- “This survey examines evaluation methods for large language model (LLM)-based agents in multi-turn conversational settings.”
- The survey proposes two taxonomies: one for “what to evaluate” and one for “how to evaluate,” covering task completion, response quality, user experience, memory/context retention, planning, and tool integration; and methodology categories including annotation-based, automated, hybrid, and self-judging evaluations.
