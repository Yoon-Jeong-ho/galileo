# Advances in Embodied Navigation Using Large Language Models: A Survey

- Year: 2025
- Venue: arXiv (survey)
- Authors: Jinzhou Lin, Han Gao, Xuxiang Feng, Rongtao Xu, Changwei Wang, Dong An, Jie Zhou, Man Zhang, Li Guo, Xiaoqiang Teng, Shibiao Xu
- URL: https://arxiv.org/html/2311.00530v5
- BibTeX key (if we add it): lin2025embodiednavllmsurvey
- Tags: survey, embodied-navigation, LLM, planning, grounded-language, multimodal

## One-sentence takeaway

A broad survey that categorizes how LLMs are used in embodied navigation (as grounded language understanding modules vs planners), reviews datasets/benchmarks, and lists challenges like multimodal fusion, latency, and spatial reasoning.

## What problem does it solve?

- Consolidates a fast-moving literature on “LLM + embodied navigation” and provides a taxonomy of integration patterns, common task settings, and dataset landscape.
- Aims to clarify what LLMs actually contribute (semantic understanding vs decision/planning) and what remains hard for real-time embodied agents.

## What is the core method / protocol?

- Survey + taxonomy (not a new algorithm).
- Organizes prior work along roles for LLMs:
  - **Information/grounding role:** LLM extracts goal-relevant semantic info from text/vision outputs; downstream exploration / control policy executes.
  - **Planner role:** LLM directly produces high-level action sequences or plans (often via dialog/prompting), which are executed by a navigation policy/controller.
- Includes discussion of multimodal extensions (vision-language models, sensor fusion) and contrasts with classic VLN (non-LLM) approaches.

## What are the key metrics?

- As a survey, metrics are those used in constituent embodied navigation tasks/benchmarks (e.g., success rate / SPL in navigation; task completion; efficiency; sometimes language grounding accuracy), but the paper itself does not introduce a unified new metric.

## What are the main results?

- Qualitative synthesis: LLM-based agents can improve semantic scene understanding and high-level planning flexibility, but face persistent issues:
  - multimodal alignment/fusion and reliance on additional vision modules,
  - **latency** and compute for real-time control,
  - weaknesses in **spatial reasoning** / fine-grained navigation,
  - data/benchmark limitations and sim-to-real gaps.

## How is this similar to GALILEO?

- Overlaps at the “LLM as reasoning/planning component” framing: LLMs used either as planners or as semantic interpreters feeding a downstream decision process.
- Highlights evaluation pain points we also face: robustness, dataset/benchmark limitations, and the need to characterize failure modes rather than only aggregate scores.

## How is this different from GALILEO?

- Domain: embodied navigation/robotics-style tasks, not our multi-round robustness / survival / turn-of-failure evaluation setting.
- Contribution type: survey and taxonomy vs GALILEO’s measurement protocol + analysis artifacts.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can offer a **more precise, auditable evaluation protocol** (survival/TOF + exported artifacts) compared to survey-level discussion.
- GALILEO’s flip/TOF analysis style could be positioned as a general methodology to diagnose “where LLM planning breaks” (including embodied settings, potentially as future work).

## Where GALILEO is weaker / needs to improve

- Survey reminds that many application domains (embodied, multimodal) stress **latency** and **multimodal grounding**; GALILEO’s current evaluation may not directly probe these axes.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, explicitly acknowledge the two LLM roles (planner vs information/grounding module) as a general pattern; tie GALILEO’s evaluation to “planner robustness over rounds.”
- [ ] Consider a short “broader impact / generality” note: our robustness diagnostics should transfer to interactive/embodied planners, but require additional observability (sensors, action traces).

## Quotes / details to potentially cite

- Abstract-level framing: LLMs enhance embodied navigation via “advanced environmental perception and decision-making support” and the survey forecasts challenges like multimodal integration and latency for real-time applications.
