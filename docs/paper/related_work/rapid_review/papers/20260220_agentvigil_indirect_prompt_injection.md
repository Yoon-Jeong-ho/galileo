# AgentVigil: Generic Black-Box Red-teaming for Indirect Prompt Injection against LLM Agents

- Year: 2025
- Venue: arXiv
- Authors: Vincent Siu; Zhe Ye; Tianneng Shi; Yuzhou Nie; Xuandong Zhao; Chenguang Wang; Wenbo Guo; Dawn Song
- URL: https://arxiv.org/abs/2505.05849
- BibTeX key (if we add it): AgentVigilSiu2025
- Tags: agents, security, prompt-injection, red-teaming, black-box

## One-sentence takeaway

AgentVigil is a black-box fuzzing framework (seed corpus + MCTS-guided mutation/selection) that finds indirect prompt injection vulnerabilities in tool-using agents and substantially improves attack success rates on AgentDojo and VWA-adv.

## What problem does it solve?

- Indirect prompt injection risk assessment for LLM agents that consume untrusted external content (webpages, reviews, emails, etc.).
- Practical constraint: many deployed agents are effectively *black-box* (closed model + opaque agent logic), making gradient/white-box methods unrealistic.
- Goal: automatically discover high-success attack strings/instructions embedded in retrieved content that reliably hijack the agent’s behavior.

## What is the core method / protocol?

- Treat indirect prompt injection discovery as *black-box fuzzing* against an agent system.
- Pipeline (high level):
  - Build a “high-quality” initial **seed corpus** of attack instructions.
  - Iteratively refine / generate candidates.
  - Use **Monte Carlo Tree Search (MCTS)** to guide seed selection / exploration (prioritizing candidates that appear to expose weaknesses).
- Evaluation targets:
  - **AgentDojo** and **VWA-adv** (public benchmarks for indirect prompt injection / web agent adversarial settings).

## What are the key metrics?

- Attack success rate (ASR) on benchmark tasks (implicitly measured as whether the agent is successfully induced to execute attacker-chosen behavior).
- Transferability / generalization across:
  - unseen tasks,
  - different underlying LLMs,
  - (partially) across defenses.

## What are the main results?

- On AgentDojo and VWA-adv, AgentVigil reportedly achieves **~71%** and **~70%** success rates (for agents based on **o3-mini** and **GPT-4o**), described as nearly doubling baseline attack methods.
- Demonstrates cross-task and cross-model transfer and some effectiveness against defenses.
- Real-world-style demonstration: injected content can mislead agents to navigate to attacker-chosen arbitrary URLs (including malicious destinations).

## How is this similar to GALILEO?

- Both care about **multi-turn agent robustness under pressure** and failure modes arising from external interactions.
- AgentVigil operationalizes an adversarial “stress testing” mindset: rather than single-pass accuracy, it probes systematic vulnerabilities under structured perturbations (here: adversarial context).
- Provides a concrete example of why agent evaluation needs to incorporate *environmental/interaction attacks*, not just benign-task success.

## How is this different from GALILEO?

- Focus is **security red-teaming (indirect prompt injection)** rather than general robustness/reliability/stability measurement (depending on GALILEO’s exact scope).
- Primary output is an *attack generation framework* (fuzzing + MCTS), not an evaluation taxonomy or reliability surface.
- Benchmarks are security-oriented (AgentDojo/VWA-adv) and success is “agent hijacked” vs “agent completes task correctly under perturbations.”

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a unified, task-agnostic evaluation protocol/metrics across multiple stress dimensions (e.g., repeated runs, semantic perturbations, tool faults), that framing may be broader and easier to compare across systems.
- GALILEO may emphasize *measurement* and *diagnostics* over discovering maximally effective attacks.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not explicitly include **indirect prompt injection** as a stressor, it risks missing an important real-world failure mode for tool-using agents.
- If GALILEO’s perturbations are mostly benign (paraphrases, noise, tool latency), it may understate adversarial brittleness.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an “untrusted context / indirect prompt injection” stress axis (even lightweight): embed adversarial instructions into retrieved snippets and measure task success + policy violations.
- [ ] Consider adapting the *fuzzing* idea as an automated “worst-case search” procedure for evaluation (not just fixed perturbation sets).
- [ ] In related work, cite AgentVigil as representative of **black-box** automated red-teaming for agent prompt injection.

## Quotes / details to potentially cite

- “We propose a generic black-box fuzzing framework, AgentVigil, designed to automatically discover and exploit indirect prompt injection vulnerabilities across diverse LLM agents.”
- “We evaluate AgentVigil on two public benchmarks, AgentDojo and VWA-adv, where it achieves 71% and 70% success rates … nearly doubling the performance of baseline attacks.”
- “Beyond benchmark evaluations, we apply our attacks in real-world environments, successfully misleading agents to navigate to arbitrary URLs, including malicious sites.”
