# BrowserArena: Evaluating LLM Agents on Real-World Web Navigation Tasks

- Year: 2025
- Venue: arXiv
- Authors: Sagnik Anupam et al. (see arXiv)
- URL: https://arxiv.org/abs/2510.02418
- BibTeX key (if we add it): browserarena2025
- Tags: web-agents, benchmark, web-navigation, evaluation, human-feedback

## One-sentence takeaway

BrowserArena proposes a live, open-web, head-to-head evaluation platform for LLM web agents with step-level human feedback, revealing recurring real-world failure modes (captchas, pop-ups, direct URL navigation).

## What problem does it solve?

- Existing web-agent evaluations are often sandboxed or artificial, missing key real-world friction (CAPTCHAs, pop-ups, site idiosyncrasies) and providing limited diagnostic signal.
- Need scalable methodology to compare agents and attribute failures to specific interaction steps.

## What is the core method / protocol?

- “Arena-style” head-to-head comparisons: run two agents on the same user-submitted live-web task and compare.
- Collect *step-level* human annotations over agent traces to identify where/why agents fail.
- Use discovered failure clusters to build targeted datasets focusing on:
  - CAPTCHA resolution
  - Pop-up/banner dismissal
  - Direct navigation to URLs
- Analyze how different underlying language models behave across these failure modes.

## What are the key metrics?

- Primarily evaluation infrastructure + human judgment signals:
  - Head-to-head preference / win-rate style comparisons (implied by Arena framing).
  - Step-level human feedback / annotations on traces (diagnostic breakdown by failure mode).
- (From abstract only) Not enough detail to list exact success metrics (task completion rate, time, steps) without reading full paper.

## What are the main results?

- Three consistent failure modes emerge from step-level annotations: CAPTCHA resolution, pop-up banner removal, and direct URL navigation.
- Different LMs exhibit different strategy diversity / brittleness on these failure modes:
  - Example: o4-mini uses a wider variety of CAPTCHA workarounds than other models.
  - Example: DeepSeek-R1 “misleads users” about whether pop-ups are closed.
- Overall conclusion: current web agents show both diversity and brittleness; step-level feedback helps characterize failures at scale.

## How is this similar to GALILEO?

- Shared theme: *evaluation protocols* that stress-test LLM/agent behavior over multi-step trajectories and try to localize failures.
- Emphasis on identifying *recurring failure modes* rather than only aggregate scores.

## How is this different from GALILEO?

- Domain: open-web navigation tasks vs GALILEO’s focus on multi-turn belief drift / sycophancy / robustness in dialogue-style settings.
- Supervision: relies on human step-level annotations and head-to-head comparisons; GALILEO aims for controlled perturbations/metrics around belief stability and revision.

## Where GALILEO is stronger / cleaner (if true)

- Can isolate causal factors (drift vs evidence-driven revision, social pressure, etc.) in a more controlled textual setting.
- Likely easier to define ground-truth “return-to-truth” style metrics than in open-web tasks with ambiguous outcomes.

## Where GALILEO is weaker / needs to improve

- Less coverage of *tool/interaction friction* failure modes (UI pop-ups, CAPTCHA-like blockers, navigation errors) that matter for deployed agents.
- May lack step-level annotations that pin failures to particular action categories.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a “step-level failure taxonomy” section for GALILEO-style multi-turn runs (e.g., classify turn-of-failure and proximate cause), inspired by BrowserArena’s annotation approach.
- [ ] For any tool-using GALILEO variants, add targeted mini-suites for common friction primitives (unexpected modal/popup, authentication gate, explicit URL-following vs search).

## Quotes / details to potentially cite

- Abstract (failure modes): “we identify three consistent failure modes: captcha resolution, pop-up banner removal, and direct navigation to URLs.”
- Abstract (method): “a live open-web agent evaluation platform … Arena-style head-to-head comparisons … step-level human feedback to surface failure modes.”
