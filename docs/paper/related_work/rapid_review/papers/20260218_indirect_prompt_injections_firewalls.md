# Indirect Prompt Injections: Are Firewalls All You Need, or Stronger Benchmarks?

- Year: 2025
- Venue: arXiv preprint
- Authors: Rishika Bhagwatkar, Kevin Kasa, Abhay Puri, Gabriel Huang, Irina Rish, Graham W. Taylor, Krishnamurthy Dj Dvijotham, Alexandre Lacoste
- URL: https://arxiv.org/abs/2510.05244
- BibTeX key (if we add it): bhagwatkar2025indirect
- Tags: prompt-injection, agents, tool-interfaces, defenses, benchmarks, evaluation

## One-sentence takeaway

A simple, model-agnostic “minimize tool inputs + sanitize tool outputs” firewall at the agent–tool boundary can saturate current indirect prompt-injection benchmarks (near-0% ASR with high utility), but the paper argues these benchmarks are too weak and need stronger attacks/metrics.

## What problem does it solve?

- Indirect prompt injection (IPI): malicious instructions embedded in external content/tool outputs that cause an LLM agent to take unintended/harmful actions or leak data.
- Need defenses that are modular/deployable without retraining and that preserve task utility.
- Need better benchmarking: current agentic security benchmarks can be “solved” by simple methods due to weak attacks and flawed metrics/bugs.

## What is the core method / protocol?

- Insert two “firewalls” at the agent–tool interface:
  - **Tool-Input Firewall (Minimizer):** reduce/strip unnecessary sensitive info from tool call arguments (mitigates data exfiltration / oversharing via tools).
  - **Tool-Output Firewall (Sanitizer):** filter tool responses before feeding them back to the agent, removing suspicious instructions/payloads.
- Designed to be **modular** and **model-agnostic**, requiring minimal assumptions and no retraining.

## What are the key metrics?

- Security: attack success rate (ASR) / “perfect security” claimed as 0% (or lowest possible) on benchmarks.
- Utility: task success rate (TSR) / maintaining task completion while defending.
- Security–utility tradeoff comparisons vs prior work.

## What are the main results?

- Reports near-0% ASR with high utility across four public benchmarks: **AgentDojo**, **Agent Security Bench**, **InjecAgent**, **tau-Bench**.
- Claims state-of-the-art security–utility tradeoff vs prior results, using a much simpler defense.
- Finds benchmark issues: flawed success metrics, implementation bugs, and especially weak attacks; proposes fixes for AgentDojo and Agent Security Bench plus best-practices.
- Notes that despite benchmark gains, **adaptive/realistic bypasses remain possible in practice**, motivating stronger attacks in evaluations.

## How is this similar to GALILEO?

- If GALILEO is concerned with safe/robust agent tool use, this is directly adjacent: it frames security as an **agent–tool boundary** problem and emphasizes **evaluation quality**.
- The “minimize & sanitize” decomposition is a clean baseline architecture for agentic safety layers.

## How is this different from GALILEO?

- Primarily a **defense + benchmark critique** paper focused on indirect prompt injection; it’s not (from the abstract/intro) proposing a new agent architecture for capability gains.
- Leans on external firewalls rather than changing the agent’s internal reasoning/training.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a principled threat model and/or stronger adaptive attacks, it can avoid the “benchmark saturation” pitfall highlighted here.
- If GALILEO integrates defenses end-to-end (not just I/O filtering), it may cover broader failure modes.

## Where GALILEO is weaker / needs to improve

- If GALILEO’s related-work section doesn’t explicitly include **tool I/O boundary defenses** and the benchmark-saturation argument, it may miss an important narrative: “simple methods already solve weak benchmarks.”
- If GALILEO evaluations rely on the same benchmarks without adaptive attacks, reviewers may question robustness.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add “minimize tool inputs + sanitize tool outputs” as a baseline/ablation in GALILEO experiments (even if only lightweight heuristics).
- [ ] In related work, explicitly cite the claim that current IPI benchmarks are saturable and discuss why GALILEO’s eval is stronger (or add stronger attacks).
- [ ] Ensure GALILEO reports both **security (ASR)** and **utility (TSR)** and discusses the measurement pitfalls called out here.

## Quotes / details to potentially cite

- “a simple, modular and model-agnostic defense operating at the agent–tool interface achieves perfect security … with high utility … across four public benchmarks …”
- Defense components: “Tool-Input Firewall (Minimizer)” and “Tool-Output Firewall (Sanitizer).”
- Benchmark critique: “critical limitations … flawed success metrics, implementation bugs, and … weak attacks …”
- Caveat: “it is still possible to bypass them in practice … need to incorporate stronger attacks in security benchmarks.”
