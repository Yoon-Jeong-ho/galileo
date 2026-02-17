# LLM-based Agents Suffer from Hallucinations: A Survey of Taxonomy, Methods, and Directions

- Year: 2025
- Venue: arXiv (cs.AI)
- Authors: Xixun Lin, Yucheng Ning, Jingwen Zhang, Yan Dong, Yilong Liu, Yongxuan Wu, Xiaohua Qi, Nan Sun, Yanmin Shang, Kun Wang, Pengfei Cao, Qingyue Wang, Lixin Zou, Xu Chen, Chuan Zhou, Jia Wu, Peng Zhang, Qingsong Wen, Shirui Pan, Bin Wang, Yanan Cao, Kai Chen, Songlin Hu, Li Guo
- URL: https://arxiv.org/abs/2509.18970
- BibTeX key (if we add it): lin2025agenthallucinations (suggested)
- Tags: agents, hallucination, survey, taxonomy, robustness

## One-sentence takeaway

A survey that frames *agent hallucinations* across the agent workflow, proposes a stage-based taxonomy, catalogs triggering causes, and summarizes detection/mitigation methods.

## What problem does it solve?

- LLM-based agents can hallucinate in ways that break task execution (e.g., wrong intermediate reasoning, incorrect tool use, fabricated observations), undermining end-to-end reliability.
- Prior work is scattered across settings (tool-using agents, long-horizon tasks, dialogue agents), making it hard to compare failure modes and mitigations.

## What is the core method / protocol?

- Survey + synthesis.
- Organizes hallucinations by *agent workflow stages* (their claimed contribution: taxonomy that localizes hallucination types to stages).
- Identifies a set of “triggering causes” (claimed: 18 causes) and groups existing approaches into hallucination detection vs mitigation.

## What are the key metrics?

- Not a single benchmark/metric paper (survey). Metrics discussed likely vary by subarea (task success, factuality/groundedness, tool-call correctness), but the abstract does not commit to a unified metric.

## What are the main results?

- Claimed contributions (from abstract):
  - “First comprehensive survey” focused specifically on hallucinations in LLM-based agents.
  - A new taxonomy tied to workflow stages.
  - An analysis of 18 triggering causes.
  - A structured review of mitigation and detection methods, plus future directions.

## How is this similar to GALILEO?

- Same high-level concern: robustness/reliability failures in *multi-step / multi-turn* systems, where errors can compound over time.
- Workflow-stage framing maps well to analyzing “when” and “how” failures occur (analogous to turn-of-failure / time-to-failure thinking, but for agent pipelines).

## How is this different from GALILEO?

- This is a survey (taxonomy + literature synthesis), not a new evaluation protocol/metric.
- Focus is hallucination broadly (including tool/observation/action errors), whereas GALILEO’s core target is multi-turn interaction robustness (and related phenomena like drift/instability/pressure).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a concrete, reproducible protocol with quantitative time-to-failure style metrics, it can offer a clearer “measurement spine” than a survey.

## Where GALILEO is weaker / needs to improve

- If our related work currently emphasizes sycophancy/pressure/consistency, we may under-cover the *agent hallucination* literature (especially tool-using settings).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add this as a related-work anchor for “hallucinations in agents” and cite their workflow-stage taxonomy as a motivating decomposition of failure points.
- [ ] Consider aligning GALILEO terminology with agent-stage failures (e.g., separate *belief/state errors* vs *action/tool errors* vs *observation/grounding errors*), even if GALILEO doesn’t evaluate tool use directly.

## Quotes / details to potentially cite

- “we present the first comprehensive survey of hallucinations in LLM-based agents.”
- “we propose a new taxonomy that identifies different types of agent hallucinations occurring at different stages.”
- “we conduct an in-depth examination of eighteen triggering causes underlying the emergence of agent hallucinations.”
- “we summarize approaches for hallucination mitigation and detection, and highlight promising directions for future research.”
