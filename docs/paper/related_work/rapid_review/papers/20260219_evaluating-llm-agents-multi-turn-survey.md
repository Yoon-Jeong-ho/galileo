# Evaluating LLM-based Agents for Multi-Turn Conversations: A Survey

- Year: 2025
- Venue: arXiv (survey)
- Authors: Shengyue Guan et al. (author list not captured from arXiv HTML via readability)
- URL: https://arxiv.org/abs/2503.22458v1
- BibTeX key (if we add it): guan2025evaluating
- Tags: multi-turn, survey, evaluation, agents, tool-use, memory, planning

## One-sentence takeaway

A survey that organizes evaluation of LLM-based multi-turn conversational agents into (i) *what* to evaluate (end-to-end, tool/action, memory, planning) and (ii) *how* to evaluate (annotation-based, automated, hybrid, LLM-as-judge), highlighting gaps around tool-use and long-term context.

## What problem does it solve?

- Evaluation for agentic, multi-turn conversational systems is fragmented (many metrics/datasets, limited coverage of tool-use and long-term context), making it hard to compare systems or design comprehensive eval suites.

## What is the core method / protocol?

- PRISMA-inspired literature review (~250 sources per abstract; larger set discussed in body) and a taxonomy.
- Two taxonomies:
  - **What to evaluate:** end-to-end experience (task completion, multitask breadth, interaction patterns, temporal aspects, UX/safety), **tool-use/action**, **memory** (span + form), and **planning**.
  - **How to evaluate:** annotation-based evaluation, automated metrics, hybrid human+automatic strategies, and self-judging (LLM-as-evaluator).

## What are the key metrics?

- Not a single new metric; summarizes commonly used ones:
  - Traditional text metrics (BLEU, ROUGE, etc.) and embedding-based (e.g., BERTScore).
  - Human judgments / rubrics for quality, coherence, usefulness.
  - Agent/task success rates and interaction-level measures.
  - Tool-use reliability / hallucination-related checks.
  - Memory retention / long-context recall style evaluations.

## What are the main results?

- Provides a structured map of evaluation targets + methods for LLM-based multi-turn agents.
- Identifies dataset/benchmark gaps, especially:
  - insufficient tool-use, multi-step, and realistic API/tool scenarios;
  - weak coverage of long-term / cross-session context maintenance;
  - difficulty capturing interactions among components (planner/tool/memory) with simple automated metrics.

## How is this similar to GALILEO?

- Aligns with GALILEO’s need to evaluate *agentic* behavior across multiple turns rather than single-turn generation.
- Explicitly calls out the same component breakdown GALILEO often uses: planning, tool-use, and memory/context.

## How is this different from GALILEO?

- This is a survey/taxonomy paper; it does not propose a concrete benchmark suite tailored to a specific system, nor a new evaluation harness.
- Focus is broader (multi-turn conversational agents generally), not specifically on GALILEO’s task distribution or operational constraints.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has an integrated evaluation harness spanning tool-use + memory + planning with reproducible tasks, it can be presented as a concrete instantiation of the “holistic evaluation” the survey argues for.

## Where GALILEO is weaker / needs to improve

- If GALILEO’s current eval is heavy on automated rubric/judge scores, this survey is a reminder to:
  - add tool-use realism (messy APIs, partial failures),
  - add long-horizon and cross-session memory tests,
  - measure interactions (planner↔tool↔memory) rather than isolated components.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, mirror this survey’s framing: “what to evaluate” vs “how to evaluate,” then position GALILEO’s eval choices accordingly.
- [ ] Add/extend benchmarks for (a) multi-tool multi-turn tasks and (b) long-term context retention (including time gaps / session boundaries if relevant).
- [ ] When using LLM-as-judge, report calibration/robustness checks (e.g., judge agreement, sensitivity to prompt, pairwise vs absolute).

## Quotes / details to potentially cite

- Abstract framing (paraphrase): introduces two taxonomies: one for *what* to evaluate (task completion, response quality, UX, memory/context, planning, tool integration) and one for *how* (annotation-based, automated, hybrid, self-judging with LLMs).
- Body emphasis (paraphrase from introduction/overview): evaluation must capture coherence/context maintenance, tool-use effectiveness, and memory management across turns.
