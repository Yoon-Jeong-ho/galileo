# STAC: When Innocent Tools Form Dangerous Chains to Jailbreak LLM Agents

- Year: 2025
- Venue: arXiv
- Authors: Jing-Jing Li; Jianfeng He; Chao Shang; Devang Kulshreshtha; Xun Xian; Yi Zhang; Hang Su; Sandesh Swamy; Yanjun Qi
- URL: https://arxiv.org/abs/2509.25624
- BibTeX key (if we add it): li2025stac
- Tags: multi-turn, tool-use, agent-safety, jailbreak, attack-chaining

## One-sentence takeaway

STAC shows that tool-using agents can be “jailbroken” via **multi-step chains of individually benign tool calls** whose harmfulness only emerges when reasoning over the **full action sequence**.

## What problem does it solve?

- Existing jailbreak/robustness work often evaluates single prompts/outputs; tool-using agents can cause real-world harm via actions.
- Safety checks that look at each step in isolation can miss **cumulative effects** (e.g., backup → delete original → delete backups).

## What is the core method / protocol?

- **Sequential Tool Attack Chaining (STAC):** construct trajectories where each tool call appears harmless locally, but the composed sequence achieves a malicious goal.
- Automated, closed-loop pipeline (as described in abstract):
  - synthesize multi-step tool chains,
  - validate malicious effect by executing in an environment,
  - “reverse-engineer” stealthy multi-turn prompts that reliably induce the agent to execute the verified sequence.
- Scale claim: 483 STAC cases; 1,352 user–agent–environment interaction sets; 10 failure modes.

## What are the key metrics?

- **Attack Success Rate (ASR)** on STAC cases (reported as >90% for many SOTA agents in the abstract).
- (Implied) coverage across domains/tasks/agent types/failure modes.

## What are the main results?

- SOTA tool-using LLM agents (incl. GPT-4.1 per abstract) are highly vulnerable to chained multi-turn tool attacks (ASR often >90%).
- Existing prompt-based defenses provide limited protection.
- A proposed **reasoning-driven defense prompt** reduces ASR by up to 28.8% (still leaving substantial residual risk).

## How is this similar to GALILEO?

- Shares the core theme: **multi-turn vulnerability** that is not captured by single-turn eval.
- Reinforces an evaluation principle relevant to GALILEO-style work: you must reason about **trajectory-level dynamics** (time/order/accumulation), not only per-turn correctness.

## How is this different from GALILEO?

- Focus is **agentic tool-use security** (environment-altering actions), not belief drift / persuasion / sycophancy per se.
- Outcome is a concrete external harm event (malicious tool execution) rather than semantic stance flips / truthfulness.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides tighter drift-vs-revision controls and recovery metrics, it can offer cleaner *behavioral diagnosis* than action-space security benchmarks.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently focuses on textual stance changes, it may under-cover **tool-mediated multi-step harm** where risk comes from composition across actions.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a related-work paragraph framing: “multi-turn failure can be **compositional** (sequence-level)”, citing STAC as the tool-use analogue.
- [ ] Consider a small appendix experiment: evaluate whether GALILEO-style monitors/metrics (e.g., time-to-failure / recovery) can be adapted to **action sequences** (e.g., first irreversible harmful action).
- [ ] In writing, emphasize that defenses should reason over **entire sequences** (aligns with our multi-turn evaluation motivation).

## Quotes / details to potentially cite

- Abstract (core claim): STAC “chains together tool calls that each appear harmless in isolation but, when combined, collectively enable harmful operations that only become apparent at the final execution step.”
- Scale + vulnerability (abstract): 483 cases; 1,352 interaction sets; ASR “exceeding 90% in most cases.”
- Defense message (abstract): “defending tool-enabled agents requires reasoning over entire action sequences and their cumulative effects, rather than evaluating isolated prompts or responses.”
