# UserBench: An Interactive Gym Environment for User-Centric Agents

- Year: 2025
- Venue: arXiv
- Authors: Cheng Qian, Zuxin Liu, Akshara Prabhakar, Zhiwei Liu, Jianguo Zhang, Haolin Chen, Heng Ji, Weiran Yao, Shelby Heinecke, Silvio Savarese, Caiming Xiong, Huan Wang
- URL: https://arxiv.org/abs/2507.22034
- BibTeX key (if we add it): userbenchQian2025
- Tags: multi-turn, agents, user-centric, preference elicitation, clarification, benchmark

## One-sentence takeaway

UserBench is an interactive benchmark with simulated users who reveal preferences gradually, measuring whether tool-using LLM agents *proactively clarify underspecified goals* and stay aligned with user intent across multi-turn interactions.

## What problem does it solve?

- Existing agent evals emphasize task completion (often with fully specified goals), but under-measure **collaborative alignment** when users start with vague/evolving goals.
- Need a way to quantify whether an agent uncovers latent preferences (via clarification) vs “charging ahead” with partially wrong assumptions.

## What is the core method / protocol?

- Benchmark environment (“interactive gym”) with **simulated users**.
- Users begin with **underspecified goals** and reveal preferences incrementally over multiple turns.
- Agent is expected to:
  - ask clarifying questions to uncover preferences/constraints
  - use tools to make grounded decisions (tool use is mentioned in the abstract; details likely include retrieval / planning actions)
  - produce a final outcome that matches *all* user intents
- Reported emphasis: gap between apparent completion and **user alignment**.

## What are the key metrics?

(From abstract; paper likely contains more formal task-specific metrics.)

- “Fully align with all user intents” rate (alignment success)
- Preference discovery / coverage: fraction of user preferences uncovered through interaction
- Task completion (implicitly contrasted with alignment)

## What are the main results?

- Across leading open/closed LLMs, full alignment with all user intents is low: **~20% on average**.
- Even the most advanced models uncover **<30%** of user preferences via active interaction.
- Highlights a disconnect: agents can look successful on completion while failing on user intent alignment.

## How is this similar to GALILEO?

- Shares the theme that **multi-turn interaction quality** matters beyond single-turn correctness.
- Emphasizes behavioral evaluation where failures can be “hidden” if you only look at end-task completion.
- Useful as adjacent evidence that *interactive robustness/alignment* is still weak even for strong LLMs.

## How is this different from GALILEO?

- UserBench focuses on **preference elicitation and collaborative task specification**, rather than (e.g.) resistance to social pressure / sycophancy / drift-vs-revision dynamics.
- Uses **simulated users** with incremental preference revelation; GALILEO’s emphasis (as used in our related-work shortlist) appears closer to pressure/robustness dynamics and failure modes under challenge.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides controlled manipulations to separate *helpful updating* vs *harmful drift* (and/or recovery dynamics), that can be cleaner diagnostically than a broad user-centric success metric.
- If GALILEO uses explicit pressure/evidence controls, it can offer more causal interpretability than preference-coverage gaps.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a “realistic collaboration” slice, it may under-address the very common real-world setting: users with **underspecified/evolving goals**.
- GALILEO could benefit from explicit measurement of whether the agent **asks the right clarifying questions** and tracks preferences/constraints over turns.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short related-work paragraph: “user-centric interactive benchmarks” (UserBench as exemplar) to motivate why multi-turn evaluation must go beyond completion.
- [ ] Consider a small auxiliary eval: preference elicitation / clarification quality under ambiguous goals (even if only a few templates), reported separately from completion.
- [ ] Consider a metric analogous to “preference coverage” for GALILEO’s setting: what latent constraints does the model explicitly surface before committing?

## Quotes / details to potentially cite

- Abstract (problem framing): agents’ ability to “**proactively collaborate with users, especially when goals are vague, evolving, or indirectly expressed**” is underexplored.
- Abstract (headline results): “answers that fully align with all user intents only **20%** of the time on average” and “advanced models uncover fewer than **30%** of all user preferences through active interaction.”
