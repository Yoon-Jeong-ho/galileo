# Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems

- Year: 2025
- Venue: arXiv (cs.MA, cs.CL)
- Authors: Shaokun Zhang; Ming Yin; Jieyu Zhang; Jiale Liu; Zhiguang Han; Jingyang Zhang; Beibin Li; Chi Wang; Huazheng Wang; Yiran Chen; Qingyun Wu
- URL: https://arxiv.org/abs/2505.00212
- BibTeX key (if we add it): zhang2025which
- Tags: agents, debugging, failure-attribution, multi-agent

## One-sentence takeaway

They introduce a dataset (Who&When) and baseline methods for attributing multi-agent LLM task failures to a specific agent and step, finding current methods/models are far from reliable—especially for pinpointing the exact failure step.

## What problem does it solve?

- Multi-agent LLM systems fail in complex ways; debugging requires identifying *which agent* and *which step* caused the failure.
- Manual failure attribution is labor-intensive and underexplored; they propose “automated failure attribution” as a distinct task.

## What is the core method / protocol?

- Dataset contribution: **Who&When**, built from failure logs of **127** LLM multi-agent systems.
- Supervision: fine-grained annotations mapping each failure to (a) the failure-responsible **agent** and (b) the decisive error **step** (“who” + “when”).
- Methods: they develop/evaluate **three** automated attribution approaches (details not fully visible from abstract; likely variants over log reasoning / classifier / prompting).
- They also test strong reasoning LLMs as attribution models.

## What are the key metrics?

- Accuracy for identifying the **failure-responsible agent**.
- Accuracy for pinpointing the **failure step**.

## What are the main results?

- Best method: **53.5%** accuracy on identifying the responsible **agent**.
- Much harder to attribute the responsible **step**: **14.2%** accuracy.
- Some methods perform **below random**.
- Even SOTA reasoning models (e.g., OpenAI o1, DeepSeek R1 per abstract) are not practically usable for this task.

## How is this similar to GALILEO?

- Both are about **understanding and improving complex LLM agentic systems** using execution traces/logs.
- The “who/when” framing aligns with analysis tooling that can support **debugging, evaluation, and ablations** for multi-component pipelines.

## How is this different from GALILEO?

- Their focus is **post-hoc blame assignment** (attribution of failures to an agent/step), while GALILEO (as a paper) is likely focused on **designing/training/controlling** agentic behavior rather than solely diagnosing failures.
- Their primary artifact is a **benchmark dataset + baselines** for failure attribution, not an improved agent architecture (from what is visible in the abstract).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides clearer causal interventions or modular evaluation, it may offer **more actionable levers** than coarse “blame” predictions (agent/step labels).

## Where GALILEO is weaker / needs to improve

- If GALILEO includes multi-agent orchestration, reviewers may ask for stronger **diagnostics** (e.g., when and where failures occur). This paper suggests that such diagnostics are non-trivial and current automatic methods are weak.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a small “failure attribution” analysis section: categorize failures by component/phase (even if heuristic), since learned attribution may be unreliable.
- [ ] If GALILEO produces logs/traces, define a consistent schema that would support future automated attribution (agent ids, step boundaries, tool calls, intermediate decisions).
- [ ] Cite this to motivate why debugging multi-agent pipelines is hard and why structured logs/evaluation matter.

## Quotes / details to potentially cite

- “Failure attribution in LLM multi-agent systems—identifying the agent and step responsible for task failures—provides crucial clues for systems debugging…” (abstract)
- “The best method achieves 53.5% accuracy in identifying failure-responsible agents but only 14.2% in pinpointing failure steps…” (abstract)
- Dataset: “Who&When … failure logs from 127 LLM multi-agent systems with fine-grained annotations…” (abstract)
