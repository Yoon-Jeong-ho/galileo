# WAREX: Web Agent Reliability Evaluation on Existing Benchmarks

- Year: 2025
- Venue: arXiv
- Authors: Su Kara; Fazle Faisal; Suman Nath
- URL: https://arxiv.org/abs/2510.03285
- BibTeX key (if we add it): warex2025
- Tags: web-agents, reliability, robustness, evaluation, fault-injection, proxy

## One-sentence takeaway

WAREX is a plug-and-play network-layer fault/attack injection proxy that turns existing web-agent benchmarks into more realistic, failure-prone evaluations and reveals large robustness drops for current agents.

## What problem does it solve?

- Current web-agent benchmarks (e.g., WebArena/WebVoyager/REAL) typically assume stable infrastructure and deterministic sites, which overestimates real-world reliability.
- Deployed web agents face common failures (timeouts, partial loads, server errors), dynamic site changes, and adversarial content (e.g., popups / injection), none of which are well-covered by many standard benchmark harnesses.

## What is the core method / protocol?

- Provide WAREX as a transparent HTTP(S) proxy layer that sits between agent and websites.
- “Split TLS” interception lets the proxy rewrite traffic and responses without changing the agent or benchmark code.
- Configurable injection policies:
  - Failure/attack type: network delay/error pages; server error codes (e.g., 500-class / rate-limit-like behaviors); JavaScript/resource delays to simulate broken pages; plus mention of adversarial manipulation/popup-style issues.
  - Frequency targeting: match by exact/regex URL; inject on k-th occurrence; every k-th; random n times.
- Also logs efficiency/cost signals by intercepting model-service calls (latency, API calls, token counts), even if the agent/benchmark doesn’t expose them.

## What are the key metrics?

- Primary: task success rate on existing benchmark tasks under injected unreliability vs default conditions.
- Secondary (when available): latency, number of model/API calls, token usage.

## What are the main results?

- Adding WAREX conditions to WebArena, WebVoyager, and REAL causes “significant drops” in task success rates for released agents, highlighting limited robustness of SOTA web agents to routine failures.
- Demonstrates a practical evaluation approach: rather than proposing a new benchmark dataset, stress-test existing ones via an add-on layer.

## How is this similar to GALILEO?

- Aligns with an evaluation philosophy of moving beyond clean, static benchmarks toward stress testing, robustness, and real-world reliability.
- Emphasizes multi-turn agent behavior under perturbations (fail at step t even if safe at step t-1), which is central to agentic system evaluation.

## How is this different from GALILEO?

- WAREX is primarily an infrastructure/proxy-based *test harness* for web agents; it does not propose a new agent design.
- Focus is specifically browser/web-task agents and network/website fault modes (and some adversarial manipulations), rather than general agent evaluation across diverse tool environments.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets broader agent behaviors (planning, tool-use correctness, safety constraints, longitudinal consistency) across domains, it can generalize beyond web browsing.
- GALILEO can provide cleaner experimental control/attribution of failure causes if it includes structured perturbation taxonomies and standardized reporting.

## Where GALILEO is weaker / needs to improve

- If GALILEO evaluations are mostly “clean-room” (stable tools, deterministic envs), it may miss reliability failures that only appear under routine infrastructure instability.
- Web-style perturbations (timeouts, flaky JS resources, transient 5xx) are easy-to-underestimate but can dominate end-to-end success in practice.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “fault injection” axis to evaluations: timeouts/delays, transient tool errors, partial results, and retries.
- [ ] Report robustness curves vs perturbation frequency (k-th occurrence / random-n), not just a single stress setting.
- [ ] Track efficiency alongside robustness (latency, tool calls, tokens) since mitigations may increase cost.
- [ ] In related work, cite WAREX as an example of *benchmark augmentation* (turning existing suites into reliability tests via an external layer).

## Quotes / details to potentially cite

- WAREX is a “plug-and-play” add-on that “integrates with existing web agent benchmarks by simulating common website failures.”
- Key benchmark critique: standard setups assume “failure-free infrastructure” and “static and closed” sites; WAREX injects realistic failures (network/server/JS) and can model adversarial manipulation.
