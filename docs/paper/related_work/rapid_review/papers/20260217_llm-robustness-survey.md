# Evaluating and Improving Robustness in Large Language Models: A Survey and Future Directions

- Year: 2025
- Venue: arXiv (survey)
- Authors: Kun Zhang; Le Wu; Kui Yu; Guangyi Lv; Dacao Zhang
- URL: https://arxiv.org/abs/2506.11111
- BibTeX key (if we add it): Zhang2025RobustnessSurvey
- Tags: survey, robustness, evaluation, ood, adversarial

## One-sentence takeaway

A broad survey that taxonomizes “LLM robustness” into adversarial robustness, OOD robustness, and robustness evaluation, with a useful paper-collection protocol and an accompanying curated repo.

## What problem does it solve?

- Lack of a unified terminology/taxonomy for “robustness” work in LLMs (spanning prompt noise, long context, attacks/defenses, OOD/generalization, hallucination, and evaluation methodology).
- Provides an organizing framework to navigate a fast-growing literature.

## What is the core method / protocol?

- A **survey taxonomy** organized primarily by *type of perturbation* / scenario:
  - **Adversarial robustness** (noise prompts, long context, toxic prompts/attacks; plus “noise decoding” such as search/CoT/rethinking).
  - **OOD robustness** (OOD detection; PEFT as the typical adaptation pathway; hallucination/knowledge update issues).
  - **Robustness evaluation** (datasets, metrics, protocols; plus leakage concerns).
- Describes a **collection & filtering protocol** (keywords + target venues; title/abstract filtering; clustering by keywords/categories).
- Offers a formalized definition/objective sketch that mixes (i) performance under clean/perturbed inputs and (ii) a consistency distance term (e.g., KL divergence).

## What are the key metrics?

- As a survey, it does not introduce a single canonical new metric; it emphasizes:
  - **Performance degradation** under perturbations.
  - **Consistency** via distributional distance measures (mentions KL-divergence-style terms).
  - Standard evaluation themes across subareas (OOD detection, hallucination, safety/toxicity, long-context performance).

## What are the main results?

- Primary “results” are organizational:
  - A three-part robustness topology (adversarial / OOD / evaluation) and a mapping of representative methods within each.
  - Highlights that LLM robustness differs from classical ML robustness due to: unified prompt interface, generative decoding, PEFT-centric adaptation, and agent/embodied settings.
  - Provides a curated companion repo: https://github.com/zhangkunzk/Awesome-LLM-Robustness-papers

## How is this similar to GALILEO?

- Overlaps in the **robustness-evaluation** emphasis (datasets/metrics/protocols) and in viewing robustness as stability under perturbations.
- Useful as a “positioning” citation that robustness is a broad, recognized problem in LLM deployments (agents, real-world interactions).

## How is this different from GALILEO?

- This is a **broad survey**, not a targeted benchmark for multi-turn *social pressure / sycophancy / persuasion*.
- Does not focus on **multi-turn time-to-failure / survival-style metrics** or **recovery-after-flip** trajectories.
- Does not cleanly separate **evidence-driven belief revision** vs **pressure-driven drift** (a key GALILEO framing).

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can contribute *sharper causal-style controls* (neutral vs pressure; evidence vs no-evidence) and trajectory metrics (flip timing, oscillation, recovery).
- GALILEO’s domain focus (social pressure/persuasion/belief stability) is more specific than the survey’s umbrella categories.

## Where GALILEO is weaker / needs to improve

- As a survey, this paper provides **breadth** and a “big-picture” robustness taxonomy; GALILEO should ensure its related-work narrative connects to that broader robustness framing.
- GALILEO may need a clear mapping from its metrics to widely-recognized robustness terms (performance, consistency, reliability; OOD vs adversarial).

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, add a short paragraph situating GALILEO within “LLM robustness” taxonomies (adversarial vs OOD vs evaluation), then narrow to *multi-turn social-pressure robustness*.
- [ ] Consider reusing survey language (“performance/consistency/reliability”) to describe our outcomes and controls.
- [ ] Add a sentence + citation pointing to the companion “Awesome LLM robustness” repo as a breadth reference, while emphasizing our niche contribution.

## Quotes / details to potentially cite

- Survey taxonomy: “Adversarial Robustness… OOD Robustness… Evaluation of Robustness (datasets, metrics, tools).”
- Mentions robustness needs in “Agents, Embodied Intelligence” application settings.
- Companion curated list: https://github.com/zhangkunzk/Awesome-LLM-Robustness-papers
