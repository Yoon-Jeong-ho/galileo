# BrowserArena: Evaluating LLM Agents on Real-World Web Navigation Tasks

- Year: 2025
- Venue: arXiv
- Authors: Sagnik Anupam, Davis Brown, Shuo Li, Eric Wong, Hamed Hassani, Osbert Bastani
- URL: https://arxiv.org/html/2510.02418v2
- BibTeX key (if we add it): browserarena2025anupam
- Tags: agents, web, benchmark, evaluation, human-feedback, pairwise-comparison

## One-sentence takeaway

BrowserArena is a live, open-web, Chatbot-Arena-style benchmark for browser agents that uses pairwise model comparisons plus step-level human annotations to identify recurring web-interaction failure modes (captcha, pop-ups, direct navigation).

## What problem does it solve?

- Existing web-agent benchmarks are often (a) self-hosted/simulated, (b) heavily ground-truthed/static, or (c) focused on “search QA” rather than real interactive navigation.
- Ground-truth success criteria are costly to engineer and constrain task diversity; they also obscure *how* agents fail mid-trajectory.
- Need an evaluation method that better matches real user task descriptions and can scale to new tasks without bespoke success checkers.

## What is the core method / protocol?

- A live evaluation platform where:
  - Users submit an *interactive* web task description.
  - Two models are sampled and run head-to-head as browser agents (built on BrowserUse + Playwright automation).
  - Users vote Left/Right/Tie on which agent did better (Arena-style preferences → leaderboard/ELO/Bradley–Terry style estimation).
- Additionally collects *step-level* annotations:
  - For each step in each agent trace, the submitting user marks the step as correct/incorrect relative to the step goal, and provides a brief reason when incorrect.
  - Cluster/summarize these annotations to surface recurring “failure modes”.
- Case-study followups:
  - Build targeted datasets designed to trigger specific failure modes (captcha, pop-up banners, direct navigation choices), then compare model behaviors/strategies.

## What are the key metrics?

- Pairwise preference outcomes (votes): used to compute model rankings (Bradley–Terry / ELO-style).
- Human evaluator agreement on a subset of tasks (majority agreement; inter-annotator agreement), with tie-handling sensitivity.
- “VLM-as-a-judge” agreement with human labels (judge picks Left/Right/Tie from traces + GIFs; ablations trace-only vs GIF-only).
- Failure-mode incidence rates and strategy frequencies on targeted datasets (judged via an LLM judge over traces).

## What are the main results?

- Collected 109 valid user-submitted tasks (via a Prolific study) with pairwise battles across 5 models.
- Identified three prominent failure modes from step-level feedback:
  - Captcha resolution / getting stuck or needing circumvention tactics.
  - Pop-up/banner closure (e.g., privacy/cookie banners blocking interaction).
  - Direct navigation choices (going straight to a presumed relevant URL vs searching first), which can hurt efficiency/success.
- VLM judging is imperfect here:
  - There is a notable gap between human preferences and VLM-judge preferences; trace-only judging can outperform trace+GIF for at least one judge model in their experiments.

## How is this similar to GALILEO?

- Both care about *realistic* evaluation of agentic behaviors (not just final-answer QA correctness).
- Emphasizes diagnosing *failure modes* rather than only reporting aggregate success.
- Uses human feedback signals beyond binary success (step-level judgments) that can inform targeted improvements.

## How is this different from GALILEO?

- BrowserArena is primarily an *evaluation platform/benchmarking methodology* for open-web navigation tasks; it standardizes agents via BrowserUse and compares *models*.
- The core outcome is preference-based ranking + failure-mode analysis, not necessarily optimizing a specific agent architecture.
- Relies on user-submitted tasks and annotations (human-in-the-loop evaluation) rather than fully automated scoring.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides more controlled, reproducible evaluation (e.g., stable environments, deterministic resets), it may avoid the inherent non-stationarity of the live web.
- If GALILEO has clearer success definitions for its target domain, it may yield more objective metrics than preference voting.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not yet incorporate step-level human feedback, BrowserArena shows a concrete protocol for capturing and analyzing it.
- If GALILEO’s evaluation is too closed/sandboxed, BrowserArena’s live-web focus highlights important real-world blockers (captcha/popups).

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a “step-level correctness” annotation protocol for agent traces (even for a small subset) to build a failure-mode taxonomy.
- [ ] Add explicit evaluation categories for web-like friction: captcha encounters, pop-up/banner interactions, and (mis)use of direct navigation.
- [ ] If using an LLM/VLM judge anywhere, run judge ablations (trace-only vs with screenshots/GIFs) and report disagreement rates.

## Quotes / details to potentially cite

- “We introduce BrowserArena, a live open-web agent evaluation platform that collects user-submitted tasks, runs Arena-style head-to-head comparisons, and uses step-level human feedback to surface failure modes.”
- Identified failure modes: “captcha resolution, pop-up banner removal, and direct navigation to URLs.”
