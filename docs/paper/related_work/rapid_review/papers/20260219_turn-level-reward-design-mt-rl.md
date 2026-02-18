# Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Reward Design

- Year: 2025
- Venue: arXiv
- Authors: Quan Wei; Siliang Zeng; Chenliang Li; William Brown; Oana Frunza; Wei Deng; Anderson Schneider; Yuriy Nevmyvaka; Yang Katie Zhao; Alfredo Garcia; Mingyi Hong
- URL: https://arxiv.org/html/2505.11821v2
- BibTeX key (if we add it): wei2025turnlevel (suggested)
- Tags: RL, PPO, GRPO, multi-turn, agents, reward-design, search

## One-sentence takeaway

Designing and assigning rewards at the *turn* level (rather than only at trajectory end) makes multi-turn RL training for tool-using LLM agents markedly more stable and accurate, with MT-PPO providing scalable credit assignment.

## What problem does it solve?

- Multi-turn/tool-using LLM agents are typically trained with sparse outcome-only rewards (correct final answer), which causes poor credit assignment across intermediate tool calls and unstable RL training.
- Prior work sometimes merges intermediate signals into a single trajectory-level reward; this still yields coarse advantages and weak supervision for early turns.

## What is the core method / protocol?

- Formalize agent interaction as a turn-level MDP with states = interaction history and actions = per-turn model outputs (often including tool calls + tool feedback).
- Extend two common RL algorithms to explicitly incorporate turn-level intermediate rewards:
  - **MT-GRPO**: derive turn-level advantages for GRPO using intermediate + future outcome advantages; shown to help in a simple 2-turn setting but scales poorly (requires exponentially many rollouts with horizon).
  - **MT-PPO**: use PPO with a critic + GAE, assigning non-zero rewards to the *last token of each turn* (intermediate) and the last token of the whole trajectory (outcome), enabling fine-grained credit assignment without exponential rollouts.
- Case study: reasoning-augmented **search agent** (multi-turn loop of: reason → form query → retrieve passages → update context → decide stop → answer).
- Turn-level reward design in the search agent includes two families:
  - **Verifiable rewards** (rule-based): exact match answer, strict format checks, retrieval contains gold answer, and a penalty for excessive search calls.
  - **LLM-as-judge rewards**: a strong judge model scores each turn with rubric-based criteria (format, reasoning quality, search effectiveness) and includes a search-usage penalty.

## What are the key metrics?

- Answer correctness (exact match) on QA datasets.
- Format correctness (strict tag/structure compliance).
- Retrieval correctness / whether retrieved content contains accepted answer.
- Training stability indicators: reward curve variance, collapse/crash frequency, convergence speed.

## What are the main results?

- Adding well-designed **turn-level rewards** yields:
  - Higher final answer accuracy vs PPO/GRPO variants trained with outcome-only or merged trajectory rewards.
  - Much higher (near-perfect) **format correctness**.
  - More stable training dynamics and faster early convergence.
- MT-PPO is positioned as the practical choice: scalable and robust due to critic-based advantage estimation; MT-GRPO illustrates the concept but has strong computational/fixed-turn limitations.

## How is this similar to GALILEO?

- Same core theme: improving long-horizon agent behavior via *denser, more local supervision* (process/step/turn feedback) instead of only end outcomes.
- Explicitly grapples with credit assignment for tool-augmented multi-turn reasoning, which is central to agentic optimization.

## How is this different from GALILEO?

- Focuses on **RL training** (PPO/GRPO variants) for a search-agent benchmark, rather than GALILEO’s specific methodology/architecture (as currently framed in our paper).
- Rewards are primarily engineered around QA/search correctness + formatting; may not cover broader agent objectives (safety, constraint satisfaction, cost/latency tradeoffs) unless added.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s evaluation/learning signals are model-agnostic or avoid heavy RL instability, we can claim cleaner applicability than PPO-style training pipelines that can crash/collapse.
- If GALILEO provides a more general process-signal scheme beyond search (e.g., for diverse tools/environments), that’s broader than this paper’s case study.

## Where GALILEO is weaker / needs to improve

- This paper provides a concrete, well-motivated *turn-level reward assignment recipe* (incl. format + search-count penalties) and clear empirical evidence on stability; GALILEO should be explicit about comparable granular feedback signals (what, when, and how assigned).
- If GALILEO does not include explicit penalties for tool overuse / degenerate loops, this is a gap.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related work paragraph: cite as evidence that **turn-/step-level rewards** improve stability and correctness for multi-turn tool agents; contrast “merged trajectory reward” vs “assigned at turn boundaries”.
- [ ] Consider adding/ablation-testing an explicit **tool-usage penalty** (search-count / tool-call count / cost) as a simple stabilizer against degenerate over-tooling.
- [ ] If we discuss GRPO-style methods, mention the paper’s note that GRPO baselines can crash in this setting and that MT-GRPO scales poorly with horizon.

## Quotes / details to potentially cite

- They frame intermediate signals as “feedback at the granularity of a single turn in multi-turn tasks” and argue sparse outcome-only reward is inadequate for long-horizon interaction.
- MT-GRPO limitation: computing intermediate advantages can require **G^(K-1)** rollouts over horizon K (exponential) and assumes fixed number of turns across samples.
- MT-PPO turn-level reward assignment: assign reward to the last token of each intermediate turn and to the last token of the entire trajectory (enabling GAE-based credit assignment).
