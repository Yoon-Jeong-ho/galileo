# Pure Vision Language Action (VLA) Models: A Comprehensive Survey

- Year: 2025
- Venue: arXiv (survey)
- Authors: Dapeng Zhang, Jin Sun, Chenghui Hu, Xiaoyan Wu, Zhenlong Yuan, Rui Zhou, Fei Shen, Qingguo Zhou
- URL: https://arxiv.org/html/2509.19012v1
- BibTeX key (if we add it): Zhang2025PureVLA
- Tags: survey, vla, embodied, robotics

## One-sentence takeaway

A broad (300+ paper) survey that proposes a taxonomy of **pure VLA** methods by *action-generation strategy* (autoregressive / diffusion / RL / hybrid / specialized) and summarizes datasets, simulators, hardware, and open challenges.

## What problem does it solve?

- Consolidates a fast-moving VLA space where “VLMs + policies” are often reviewed only as a subtopic of either (i) VLM foundation models or (ii) general robotics manipulation surveys.
- Provides a reader-friendly map of **design choices for turning vision-language models into action-producing agents** (not just captioners/planners).
- Centralizes pointers to **resources** (datasets, benchmarks, simulators) commonly needed to make VLA work reproducible.

## What is the core method / protocol?

- This is a survey, not a new algorithm.
- Main contribution is a taxonomy and narrative structure:
  - Categorize “pure VLA” methods into: **autoregression-based**, **diffusion-based**, **reinforcement-based**, **hybrid**, and **specialized** methods.
  - Discuss motivations/strategies/implementations at a high level for each bucket.
  - Summarize application deployments across robot form factors (arms, quadrupeds, humanoids, wheeled/autonomous vehicles).
  - Summarize **datasets/benchmarks** and **simulation platforms** used by the community.
  - Identify open issues: data limitations, inference speed, safety.

## What are the key metrics?

- Survey does not introduce a single unified metric; it points to task/benchmark-driven evaluation.
- Typical evaluation axes implied by the survey scope:
  - Task success / completion rates on manipulation/nav suites.
  - Generalization: novel objects, novel instructions, novel environments.
  - Data efficiency / scaling behavior (size of demonstrations, internet-scale VLM pretraining vs robotics finetuning).
  - Efficiency: inference latency / action-rate constraints.
  - Safety: constraint satisfaction, hazardous action avoidance (often under-specified in many benchmarks).

## What are the main results?

- Produces an organizing view of the field and a bibliography-scale synthesis (“over three hundred recent studies”).
- Highlights that “pure VLA” can be understood primarily through the **action generation mechanism** (AR vs diffusion vs RL etc.) rather than only by which VLM/LLM backbone is used.

## How is this similar to GALILEO?

- If GALILEO is positioning itself around *generalization/robustness/evaluation protocols* for agents, this paper is useful as:
  - A taxonomy reference to situate the “agent/policy” component among AR/diffusion/RL/hybrid families.
  - A pointer list to common datasets/simulators that reviewers expect you to mention when claiming coverage of embodied/agent-like behavior.

## How is this different from GALILEO?

- This is a **broad survey**; it does not propose a new evaluation protocol, metric suite, or controlled experimental design.
- It is robotics/VLA-focused; it does not center the “multi-turn robustness / drift / recovery” type of evaluation story (if that is GALILEO’s focus).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides **clean controlled protocols + metrics** (e.g., separating failure modes, reporting time-to-failure/recovery, or stronger ablations), that level of methodological precision is typically beyond what a survey can provide.

## Where GALILEO is weaker / needs to improve

- If GALILEO touches embodied/VLA-adjacent claims, this survey suggests a bar for:
  - Clear positioning w.r.t. **AR vs diffusion vs RL** action-generation paradigms.
  - Explicit listing of **datasets/benchmarks/simulators** that define the space.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, explicitly name the “pure VLA” taxonomy dimension: action-generation strategy (AR/diffusion/RL/hybrid), and state where GALILEO fits (or why it is orthogonal).
- [ ] Use this paper as a citation hub to justify a compact background paragraph on VLA and why “VLMs as agents” is a paradigm shift (sequence generator -> active decision-maker).
- [ ] If we make any embodied/agent generalization claims, cross-check that we mention at least the standard datasets/simulators referenced by this survey (even if we do not run them).

## Quotes / details to potentially cite

- “We refer to these as VLA foundation models…” (framing VLA as foundation models for manipulation).
- Taxonomy statement: VLA approaches categorized into “autoregression-based, diffusion-based, reinforcement-based, hybrid, and specialized methods.”
- Scope claim: “By synthesizing insights from over three hundred recent studies…”
