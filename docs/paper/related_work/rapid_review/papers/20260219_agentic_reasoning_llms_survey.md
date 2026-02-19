# Agentic Reasoning for Large Language Models

- Year: 2026
- Venue: arXiv (survey)
- Authors: Tianxin Wei, Ting-Wei Li, Zhining Liu, Xuying Ning, Ze Yang, Jiaru Zou, Zhichen Zeng, Ruizhong Qiu, Xiao Lin, Dongqi Fu, Zihao Li, Mengting Ai, Duo Zhou, Wenxuan Bao, Yunzhe Li, Gaotang Li, Cheng Qian, Yu Wang, Xiangru Tang, Yin Xiao, Liri Fang, Hui Liu, Xianfeng Tang, Yuji Zhang, Chi Wang, Jiaxuan You, Heng Ji, Hanghang Tong, Jingrui He
- URL: https://arxiv.org/abs/2601.12538
- BibTeX key (if we add it): wei2026agenticreasoning
- Tags: agents, survey, reasoning, planning, tool-use, multi-agent, post-training

## One-sentence takeaway

A broad 2026 survey that frames **agentic reasoning** as LLMs that plan/act/learn via interaction, organized by (i) environment dynamics (foundational → self-evolving → multi-agent) and (ii) learning regime (in-context orchestration vs post-training).

## What problem does it solve?

- Consolidates a rapidly growing and fragmented literature on “LLM-as-agent” reasoning into a structured taxonomy.
- Highlights gaps when moving from closed-world reasoning to **open-ended, dynamic, interactive** environments.

## What is the core method / protocol?

- Not a new algorithm; a **survey + taxonomy**.
- Organizes work along three layers of “environmental dynamics”:
  - **Foundational agentic reasoning**: single-agent planning, tool use, search in relatively stable environments.
  - **Self-evolving agentic reasoning**: feedback, memory, adaptation over time.
  - **Collective multi-agent reasoning**: coordination, knowledge sharing, shared goals.
- Cross-cuts these with:
  - **In-context reasoning** (test-time orchestration / prompting / structured interaction).
  - **Post-training reasoning** (SFT/RL to shape agent behaviors).

## What are the key metrics?

- Not standardized in the paper (survey), but it points to common evaluation families:
  - Task success / accuracy in interactive settings.
  - Tool-use correctness and efficiency.
  - Long-horizon performance / robustness under environment dynamics.
  - Multi-agent coordination outcomes.

## What are the main results?

- Main “result” is a **unified roadmap** + a list of open challenges (personalization, long-horizon interaction, world modeling, scalable multi-agent training, governance).
- Provides an “Awesome” companion resource (curation) which can be useful for ensuring related-work coverage.

## How is this similar to GALILEO?

- Shares the high-level framing that **interaction** changes what “reasoning” means (thought tightly coupled to action, feedback, and longer horizons).
- Useful as a citation umbrella when motivating why static, closed-world reasoning benchmarks are insufficient.

## How is this different from GALILEO?

- This is a **survey**, not a concrete benchmark/protocol or a specific intervention.
- It is broad across domains (science/robotics/healthcare/math/autonomous research) rather than tightly centered on one failure mode or one evaluation harness.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO contributes a tight experimental protocol + metrics (e.g., drift vs revision, recovery dynamics, pressure/evidence controls), it will be **more falsifiable and reviewer-auditable** than survey taxonomies.

## Where GALILEO is weaker / needs to improve

- Coverage risk: surveys like this can make omissions in related work more visible; GALILEO should ensure its positioning uses a comparable vocabulary (agentic reasoning, in-context vs post-training, etc.).

## Action items for GALILEO (experiments / method / writing)

- [ ] Use this as a **framing citation** for “agentic reasoning = planning+acting+learning in dynamic environments” (especially in intro/related-work lead-in).
- [ ] Mirror the survey’s cross-cut vocabulary (in-context vs post-training) when describing which parts of the problem GALILEO addresses.
- [ ] Skim the companion curated list to ensure we did not miss any must-cite interactive agent benchmarks/frameworks adjacent to our claims.

## Quotes / details to potentially cite

- Abstract-level definition: agentic reasoning reframes LLMs as autonomous agents that “plan, act, and learn through continual interaction.”
- Taxonomy hooks: “foundational / self-evolving / collective multi-agent” + “in-context vs post-training.”
