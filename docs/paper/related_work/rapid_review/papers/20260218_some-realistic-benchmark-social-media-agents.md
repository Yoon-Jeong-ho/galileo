# SoMe: A Realistic Benchmark for LLM-based Social Media Agents

- Year: 2025
- Venue: AAAI 2026 (accepted)
- Authors: Dizhan Xue; Jing Cui; Shengsheng Qian; Chuanrui Hu; Changsheng Xu
- URL: https://arxiv.org/abs/2512.14720
- BibTeX key (if we add it): some2025
- Tags: social-media, agents, tool-use, multi-task, benchmark, evaluation

## One-sentence takeaway

SoMe is a large-scale, tool-augmented benchmark for evaluating LLM-based *social media agents* across 8 realistic tasks, and it finds that mainstream open/closed models still struggle in this setting.

## What problem does it solve?

- Existing evaluations of “social media agents” tend to be narrow (single-task), small-scale, or lack ground-truth references.
- There is no comprehensive, realistic testbed that (i) exposes agents to real social media data + external sources, (ii) requires tool use, and (iii) spans multiple socially grounded tasks.

## What is the core method / protocol?

- Benchmark + platform for social-media-agent evaluation:
  - 8 task families: real-time event detection; streaming event summarization; misinformation detection; user behaviour prediction; user emotion analysis; user comment simulation; media content recommendation; social media question-answering.
  - A suite of “agent tools” for acquiring/managing/analyzing social media data (tool-using agents).
- Large-scale data + annotations:
  - ~9.16M posts, 6,591 user profiles, 25,686 reports, collected from many platforms + external websites.
  - 17,869 annotated task queries.
- Evaluation is a mixture of quantitative + qualitative analysis; the paper mentions using an LLM scorer to assist evaluation (details likely in full paper).

## What are the key metrics?

- Task performance per task type (not fully visible from abstract-only skim).
- LLM-judge assisted scoring (“LLM scorer” for answers), plus qualitative error analysis / limitation taxonomy.

## What are the main results?

- “Both the current closed-source and open-source LLMs cannot handle social media agent tasks satisfactorily” (abstract).
- Provides a first broad snapshot of failure modes / limitations for agentic LLMs in realistic social-media environments.

## How is this similar to GALILEO?

- Same general shape: evaluate *agentic* LLM behavior in complex, multi-step, tool-mediated environments.
- Emphasis on realism and robustness: measuring how systems behave when decisions require integrating heterogeneous evidence (posts, profiles, external reports).

## How is this different from GALILEO?

- SoMe is domain-specific (social media) and multi-task; GALILEO’s focus is broader (general interaction robustness / drift / stability, depending on our framing).
- SoMe is primarily a benchmark + dataset + tool platform; not explicitly centered on belief revision vs drift controls, survival-style robustness metrics, or recovery interventions.

## Where GALILEO is stronger / cleaner (if true)

- Opportunity: clearer causal/protocol separation between (a) evidence-driven revision vs (b) social pressure / drift, with explicit controls and turn-level metrics.
- Opportunity: more principled longitudinal metrics (time-to-failure, recovery curves) vs aggregate task scores.

## Where GALILEO is weaker / needs to improve

- SoMe’s realism at scale (millions of posts; thousands of profiles; multi-platform) is hard to match.
- SoMe’s breadth of social tasks suggests we may be under-covering “social” agent tasks beyond persuasion/sycophancy (e.g., recommendation, event summarization, comment simulation).

## Action items for GALILEO (experiments / method / writing)

- [ ] Related-work: cite SoMe as a large-scale, realistic benchmark for tool-using social media agents; use it to motivate why today’s agentic LLMs still fail outside toy settings.
- [ ] Consider adding (or explicitly contrasting with) a “realistic tool+data” condition: agents operating over multi-source social data, then evaluate stability/robustness metrics we care about.
- [ ] If we use LLM-judge scoring, be explicit about judge robustness / calibration; SoMe’s use of LLM scorer is a good comparison point.

## Quotes / details to potentially cite

- “SoMe comprises a diverse collection of 8 social media agent tasks, 9,164,284 posts, 6,591 user profiles, and 25,686 reports … with 17,869 meticulously annotated task queries.” (abstract)
- “Our evaluation reveals that both the current closed-source and open-source LLMs cannot handle social media agent tasks satisfactorily.” (abstract)
