# Adaptation of Agentic AI

- Year: 2025
- Venue: arXiv (survey/framework)
- Authors: Pengcheng Jiang, Jiacheng Lin, Zhiyi Shi, Zifeng Wang, Luxi He, Yichen Wu, Ming Zhong, Peiyang Song, Qizheng Zhang, Heng Wang, Xueqiang Xu, Hanwen Xu, Pengrui Han, Dylan Zhang, Jiashuo Sun, Chaoqi Yang, Kun Qian, Tian Wang, Changran Hu, Manling Li, Quanzheng Li, Hao Peng, Sheng Wang, Jingbo Shang, Chao Zhang, Jiaxuan You, Liyuan Liu, Pan Lu, Yu Zhang, Heng Ji, Yejin Choi, Dawn Song, Jimeng Sun, Jiawei Han
- URL: https://arxiv.org/abs/2512.16301
- BibTeX key (if we add it): Jiang2025AdaptationAgenticAI
- Tags: agents, adaptation, tool-use, survey, framework

## One-sentence takeaway

A unifying framework for “adaptation” in agentic AI that separates *agent adaptation* vs *tool adaptation* and further classifies signals/supervision, aiming to make design trade-offs explicit.

## What problem does it solve?

- The literature on improving agentic systems is sprawling (prompting, fine-tuning, memory, tool-use optimization, etc.), making it hard to reason about what is being adapted, with what signals, and what trade-offs follow.
- Provides a taxonomy so practitioners can choose/switch adaptation strategies during system design.

## What is the core method / protocol?

- Conceptual framework (survey):
  - **Agent adaptations** vs **tool adaptations**.
  - Agent adaptation is decomposed into:
    - **Tool-execution-signaled** (adapt based on tool-call outcomes/feedback).
    - **Agent-output-signaled** (adapt based on the agent’s own produced outputs).
  - Tool adaptation is decomposed into:
    - **Agent-agnostic** (optimize tools/interfaces independent of a specific agent).
    - **Agent-supervised** (optimize tools guided by agent behavior/feedback).
- Reviews representative methods within each bucket; discusses strengths/limitations and open challenges.

## What are the key metrics?

- Not a benchmark paper; evaluation is via qualitative comparison across categories.
- Useful “metrics” to track for a system (implied by the survey framing): reliability of tool execution, generalization to new tasks/tools, data/compute cost of adaptation, and safety/robustness under distribution shift.

## What are the main results?

- A structured map of adaptation strategies for agentic AI and tool-use systems.
- A set of design considerations/trade-offs and open problems (e.g., when to adapt agent vs tool; what signals are trustworthy; supervision cost; stability vs plasticity; safety constraints).

## How is this similar to GALILEO?

- Directly overlaps with the “agent + tools + feedback loops” viewpoint.
- The paper’s taxonomy can be used to position GALILEO’s training/inference choices (e.g., whether improvements come from modifying the agent policy vs improving tool interfaces/executors).

## How is this different from GALILEO?

- This work is a **survey/framework**, not a new algorithm or empirical system.
- It does not provide a concrete implementation recipe or quantitative evidence for a specific adaptation mechanism; instead it organizes prior work.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes a single, end-to-end-defined objective/protocol and quantitative evaluation, it will read as more concrete than this taxonomy paper.

## Where GALILEO is weaker / needs to improve

- Writing risk: without an explicit positioning, GALILEO could be perceived as “another adaptation variant”; using this framework could help clarify *what* is adapted and *what signals* drive the adaptation.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work / method framing, explicitly classify GALILEO along this paper’s axes:
  - agent adaptation vs tool adaptation
  - tool-execution-signaled vs agent-output-signaled
  - agent-agnostic vs agent-supervised tool adaptation
- [ ] Add a short paragraph discussing the trade-off GALILEO chooses (e.g., supervision cost vs reliability; stability vs plasticity).

## Quotes / details to potentially cite

- Abstract framing: “adaptation becomes a central mechanism for improving performance, reliability, and generalization.”
- Abstract contribution: “unify the rapidly expanding research landscape into a systematic framework that spans both agent adaptations and tool adaptations.”
- Category names (from abstract): “tool-execution-signaled and agent-output-signaled… agent-agnostic and agent-supervised…”
