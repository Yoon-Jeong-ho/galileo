# Where LLM Agents Fail and How They can Learn From Failures

- Year: 2025
- Venue: arXiv
- Authors: Kunlun Zhu, Zijia Liu, Bingxuan Li, Muxin Tian, Yingxuan Yang, Jiaxun Zhang, Pengrui Han, Qipeng Xie, Fuyang Cui, Weijia Zhang, Xiaoteng Ma, Xiaodong Yu, Gowtham Ramesh, Jialian Wu, Zicheng Liu, Pan Lu, James Zou, Jiaxuan You
- URL: https://arxiv.org/abs/2509.25370
- BibTeX key (if we add it): zhu2025agenterror
- Tags: agents, failures, taxonomy, dataset, debugging, tool-use

## One-sentence takeaway

A modular taxonomy + annotated failure trajectories + a root-cause-isolating “debugger” feedback loop can measurably improve LLM agent recovery and success on tool-use benchmarks.

## What problem does it solve?

- LLM agents have multiple interacting modules (planning, memory, reflection, tool/action execution), so a single early mistake can cascade; existing evaluations often report only task success without a modular/systemic account of *why* failure occurred.
- There is no widely used, systematically annotated dataset of agent failure trajectories with root-cause labels spanning modules.

## What is the core method / protocol?

- **AgentErrorTaxonomy:** a modular classification of agent failure modes spanning memory, reflection, planning, action, and system-level operations.
- **AgentErrorBench:** a dataset of agent rollouts with systematic error annotations, built from ALFWorld, GAIA, and WebShop trajectories.
- **AgentDebug:** a debugging framework intended to:
  - identify / isolate the *root-cause* failure in a trajectory (not just the final wrong step), and
  - generate targeted corrective feedback that enables iterative recovery.

(From the abstract, AgentDebug is positioned as a general “debugger” that can be applied to existing agent trajectories; details of the root-cause isolation algorithm are not visible from the arXiv abstract alone.)

## What are the key metrics?

- **All-correct accuracy** (trajectory-level; implies all steps correct).
- **Step accuracy** (per-step correctness across a trajectory).
- **Task success** improvements from iterative recovery (reported across ALFWorld, GAIA, WebShop).

## What are the main results?

- On AgentErrorBench, AgentDebug reports:
  - **+24%** (absolute, per abstract wording) higher **all-correct accuracy** vs strongest baseline.
  - **+17%** higher **step accuracy** vs strongest baseline.
- The targeted feedback enables iterative recovery, yielding up to **26% relative** improvements in task success across ALFWorld, GAIA, WebShop.

## How is this similar to GALILEO?

- Shares the overarching view that **multi-step, multi-turn systems fail via cascades**, and that diagnostics should attribute errors to *specific causes* rather than only reporting end outcomes.
- Emphasizes **recovery** (not just failure detection), aligning with a “pressure/failure then correction” narrative.
- Supports a methodology where we can talk about **failure mode categories** and **trajectory-level outcomes**.

## How is this different from GALILEO?

- Target domain is **tool-use agents** and task benchmarks (ALFWorld/GAIA/WebShop) rather than social-pressure / belief-drift / sycophancy style dialogue dynamics.
- Primary contribution is a **debugging framework + taxonomy + annotated bench** rather than a new interaction protocol for distinguishing evidence-driven revision vs pressure-driven drift.
- Uses (at least in evaluation framing) stepwise correctness metrics rather than stance/flip/recovery metrics that GALILEO likely foregrounds.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO focuses on black-box behavioral protocols for drift vs revision (and recovery), it may provide **cleaner causal controls** specific to *social pressure* mechanisms than a general tool-use agent benchmark.

## Where GALILEO is weaker / needs to improve

- GALILEO may lack a **modular, reusable taxonomy** and a **publicly reusable annotated trajectory bench** that supports systematic error attribution.
- If we do not report **step-level diagnostics** (or an analogue), we may be missing an easy-to-communicate “where the trajectory went wrong” story.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short related-work paragraph positioning GALILEO relative to **agent debugging / root-cause isolation** (AgentDebug) and clarify what we do instead (e.g., protocol-level causal controls).
- [ ] Consider defining a lightweight **GALILEOErrorTaxonomy** aligned to our modules (prompting/role, memory/context, reflection, decision/stance update, action/response), to improve reviewer-facing analysis.
- [ ] Add an “all-correct / step-accuracy analogue” for our setting (e.g., **all-turn-consistent**; **per-turn stability**), to parallel AgentDebug’s reporting.

## Quotes / details to potentially cite

- “LLM agents … amplify vulnerability to cascading failures, where a single root-cause error propagates through subsequent decisions, leading to task failure.”
- “AgentErrorTaxonomy, a modular classification of failure modes spanning memory, reflection, planning, action, and system-level operations.”
- “AgentErrorBench … systematically annotated failure trajectories from ALFWorld, GAIA, and WebShop …”
- “AgentDebug achieves 24% higher all-correct accuracy and 17% higher step accuracy …”
- “Targeted feedback … yielding up to 26% relative improvements in task success …”
