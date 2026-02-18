# AgentDyn: A Dynamic Open-Ended Benchmark for Evaluating Prompt Injection Attacks of Real-World Agent Security System

- Year: 2026
- Venue: arXiv (cs.CR)
- Authors: Hao Li, et al. (see arXiv)
- URL: https://arxiv.org/abs/2602.03117
- BibTeX key (if we add it): AgentDyn2026Li
- Tags: agents, security, prompt-injection, indirect-injection, benchmark, tool-use, evaluation

## One-sentence takeaway

AgentDyn is a manually constructed benchmark (60 open-ended tasks, 560 injection cases) intended to stress-test indirect prompt-injection defenses for tool-using agents under more realistic, dynamic, instruction-rich conditions, showing most current defenses are either insecure or overly restrictive.

## What problem does it solve?

- Existing prompt-injection/agent-security benchmarks are argued to be misaligned with real-world agent deployments.
- The paper claims three recurring benchmark gaps:
  - Tasks are too static / not truly open-ended or requiring dynamic planning.
  - Benchmarks omit “helpful” third-party instructions that occur naturally in the wild (and can be intertwined with attacks).
  - User tasks are overly simplistic.

## What is the core method / protocol?

- Introduce **AgentDyn**, a manually designed benchmark suite with:
  - **60** challenging open-ended tasks
  - **560** injection test cases
  - Domains: **Shopping**, **GitHub**, **Daily Life**
- Key design emphasis vs prior work: tasks that require multi-step planning + environments where benign third-party instructions coexist with adversarial injections.
- Evaluate **10** state-of-the-art defenses on this benchmark (defense methods not detailed on the arXiv abstract page).

## What are the key metrics?

- Security / robustness to indirect prompt injection (attack success vs defense success).
- Utility / “over-defense” (how much the defense blocks legitimate actions/helpful instructions).
- (Exact metric definitions and scoring likely in the paper/tables; not present in the abstract.)

## What are the main results?

- Across 10 defenses, the authors report a consistent tradeoff:
  - Many defenses are **not secure enough** under AgentDyn.
  - Others show **significant over-defense**, harming agent task completion.
- Conclusion: existing defenses are still far from real-world readiness for agent deployments.

## How is this similar to GALILEO?

- Both care about **realistic evaluation** for agent behavior in the presence of tool use and external content.
- Reinforces the need to explicitly test **dynamic planning + mixed benign/malicious instructions**, which is a core pain point for agentic systems.

## How is this different from GALILEO?

- AgentDyn is a **benchmark** for prompt-injection defenses; it is not (from the abstract) a new defense method.
- Focus is specifically **indirect prompt injection** in tool-using agents across a few application domains.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a clearer threat model and evaluation methodology across broader behaviors (beyond injection), it can position AgentDyn as a complementary targeted benchmark.

## Where GALILEO is weaker / needs to improve

- If GALILEO evaluation does not explicitly include:
  - open-ended tasks with dynamic planning,
  - benign “helpful” third-party instructions mixed with attacks,
  - a measured utility vs security tradeoff,
  then AgentDyn’s framing suggests gaps to address.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add/expand an evaluation slice that explicitly measures **security–utility tradeoff** (over-defense) when filtering/guarding tool calls.
- [ ] Ensure at least some tasks require **dynamic replanning** with third-party content that contains both benign instructions and adversarial injections.
- [ ] Cite AgentDyn as evidence that **current benchmarks under-stress** real-world agent security conditions.

## Quotes / details to potentially cite

- “we reveal three fundamental flaws in current benchmarks … (i) lack of dynamic open-ended tasks, (ii) lack of helpful instructions, and (iii) simplistic user tasks.”
- “AgentDyn … 60 challenging open-ended tasks and 560 injection test cases across Shopping, GitHub, and Daily Life.”
- “almost all existing defenses are either not secure enough or suffer from significant over-defense.”
