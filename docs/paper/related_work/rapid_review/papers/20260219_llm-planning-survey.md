# Large Language Models for Planning: A Comprehensive and Systematic Survey

- Year: 2025
- Venue: arXiv
- Authors: Pengfei Cao; Tianyi Men; Wencan Liu; Jingwen Zhang; Xuzhao Li; Xixun Lin; Dianbo Sui; Yanan Cao; Kang Liu; Jun Zhao
- URL: https://arxiv.org/html/2505.19683v1
- BibTeX key (if we add it): cao2025llmplanningsurvey
- Tags: survey; planning; llm

## One-sentence takeaway

A broad survey that organizes LLM-based planning into (i) external-module augmentation, (ii) finetuning with trajectories/feedback, and (iii) search/decomposition/decoding-time methods, plus evaluation resources and open challenges.

## What problem does it solve?

- Provides an up-to-date taxonomy and literature map for “LLM-based planning” (LLM agents used for sequential decision-making / long-horizon tasks), which is otherwise fragmented across symbolic planning, RL-ish finetuning, and search-based prompting.
- Summarizes how planning is defined (MDP/POMDP, open/closed loop, high/low level), what method families exist, and what benchmarks/metrics are used.

## What is the core method / protocol?

- Not a new algorithm; it is a structured survey.
- Main categorization of methods:
  - External Module Augmented Methods
    - Planner-enhanced: translate NL to formalism (e.g., PDDL / ASP / SMT / MILP), solve with external planner/verifier, possibly translate back.
    - Memory-enhanced: add explicit memory modules (experience / long-term conversation / retrieval) to improve long-horizon planning.
  - Finetuning-based Methods
    - Imitation learning: finetune on expert/self-generated/hybrid trajectories.
    - Feedback-based: finetune with preference/critic/reward signals (RLHF-style variants) for better planning behavior.
  - Searching-based Methods
    - Decomposition-based: break tasks into subproblems (tool routing / skill decomposition).
    - Exploration-based: explicit search over reasoning/planning trees (e.g., ToT/MCTS-like scaffolds).
    - Decoding-based: modify decoding to improve plan quality (contrastive/grounded/predictive decoding, etc.).
- Also claims an “awesome list” resource of 300+ papers.

## What are the key metrics?

- Survey-level: emphasizes benchmarking and evaluation metrics but specifics depend on task suites.
- Typical metrics implied by covered areas:
  - Task success rate / goal completion
  - Plan validity / constraint satisfaction (especially for formal planners)
  - Efficiency: steps, cost, time, token usage
  - Robustness to environment change / partial observability
  - Generalization across tasks/domains

## What are the main results?

- Consolidates prior work and argues:
  - Planner-enhanced methods can yield reliable plans but bottleneck on correct translation to formal languages.
  - Memory-augmented agents help long-horizon planning but depend heavily on retrieval quality and memory update strategies.
  - Finetuning improves planning but requires trajectory/feedback data and careful alignment to avoid brittleness.
  - Search/scaffolding methods often improve solution quality but add compute and can be sensitive to heuristics.

## How is this similar to GALILEO?

- If GALILEO is an agent/planning system, this survey’s framing matches common building blocks:
  - using external tools/solvers/verifiers;
  - using memory for long-horizon tasks;
  - using search/decomposition to structure multi-step behavior;
  - needing clear evaluation (success, efficiency, robustness).

## How is this different from GALILEO?

- Survey (taxonomy + literature + benchmarks), not a single method.
- Focuses broadly on “LLM planning” across domains (web/desktop/mobile/embodied/travel), whereas GALILEO likely targets a narrower setting with a concrete system design.

## Where GALILEO is stronger / cleaner (if true)

- A well-specified GALILEO pipeline with explicit state/action interfaces and reproducible evaluation would be “cleaner” than many ad-hoc agent papers aggregated here.
- If GALILEO emphasizes verifiability (e.g., constraint checks, structured outputs), it aligns with the survey’s recommendation to use external verifiers/planners.

## Where GALILEO is weaker / needs to improve

- Ensure GALILEO is positioned against all three families in the taxonomy (augmentation vs finetuning vs search/decoding), not just one.
- If GALILEO does not yet have a thorough benchmark section (datasets + metrics + baselines), this survey highlights that as a key expectation for planning papers.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, adopt the survey’s 3-way taxonomy headings (External module augmented / Finetuning-based / Searching-based) and place GALILEO explicitly.
- [ ] Add at least one baseline from each bucket (or justify exclusion), to avoid reviewer pushback that comparisons are incomplete.
- [ ] Clarify which planning setting GALILEO targets using the survey’s definitions (MDP vs POMDP; open-loop vs closed-loop; high-level vs low-level).
- [ ] If applicable, cite this survey as an umbrella reference, then cite a small set of representative methods under each category.

## Quotes / details to potentially cite

- Abstract taxonomy (three principal approaches): external-module augmented, finetuning-based, searching-based.
- Definition framing: planning as tuple (S, A, T, s_init, S_goal) and discussion of MDP vs POMDP; open-loop vs closed-loop; high-level vs low-level.
- Resource link mentioned: https://github.com/Quester-one/Awesome-LLM-Planning
