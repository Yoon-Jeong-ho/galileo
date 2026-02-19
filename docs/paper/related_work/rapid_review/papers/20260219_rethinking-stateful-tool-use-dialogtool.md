# Rethinking Stateful Tool Use in Multi-Turn Dialogues: Benchmarks and Challenges

- Year: 2025
- Venue: arXiv
- Authors: Hongru Wang; Wenyu Huang; Yufei Wang; Yuanhao Xi; Jianqiao Lu; Huan Zhang; Nan Hu; Zeming Liu; Jeff Z. Pan; Kam-Fai Wong
- URL: https://arxiv.org/abs/2505.13328
- BibTeX key (if we add it): wang2025dialogtool
- Tags: tool-use, stateful, multi-turn, benchmark, agents

## One-sentence takeaway

DialogTool + VirtualMobile provide a staged benchmark for *stateful* multi-turn tool use (from tool creation to execution to role-consistent responses), and show that current LLMs still struggle over long horizons.

## What problem does it solve?

- Existing “tool-use agent” benchmarks often evaluate tool selection/execution in isolated or single-turn settings, missing the *stateful* lifecycle of tools across multi-turn dialogues.
- As a result, measured performance can overestimate real deployment capability where tools/APIs evolve, state persists, and the assistant must stay consistent in role/outputs.

## What is the core method / protocol?

- Proposes **DialogTool**, a multi-turn dialogue dataset covering the *full lifecycle* of tool use in three stages:
  1) **Tool creation** (define/construct tools/APIs)
  2) **Tool utilization**: tool awareness, tool selection, tool execution
  3) **Role-consistent response**: response generation + role play
- Builds **VirtualMobile**, an embodied “virtual mobile” evaluation environment that simulates API calls and (importantly) tests **robustness** of created APIs/tools under varied conditions.
- Evaluates **13** open- and closed-source LLMs with stage-by-stage analysis.

## What are the key metrics?

- The abstract emphasizes staged task performance across (creation → awareness/selection/execution → role-consistent response) and robustness of created APIs in VirtualMobile.
- (Needs PDF skim for exact metric names; likely per-task success / accuracy / execution success, plus robustness stress tests in the simulator.)

## What are the main results?

- Across 13 models, “state-of-the-art LLMs still cannot perform well to use tools over long horizons,” with failures observable at multiple stages of the lifecycle.
- The paper argues that partial/stateless tool-use evaluation misses these long-horizon and state-dependent failure modes.

## How is this similar to GALILEO?

- Shares the core motivation: **multi-turn, long-horizon evaluation** that exposes degradation/failure not visible in single-turn tests.
- Emphasizes **robustness under realistic interaction structure** (state, lifecycle, sequential dependency).

## How is this different from GALILEO?

- Focuses specifically on **tool-use lifecycle** and **stateful tool interaction**, not primarily on social pressure / persuasion / stance drift dynamics.
- Introduces an **embodied simulator (VirtualMobile)** to exercise APIs; GALILEO may instead center on conversational dynamics / pressure protocols (depending on the exact GALILEO definition in the paper).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets controlled conversational pressure/drift/consistency phenomena, it may offer cleaner causal knobs and simpler task-independent metrics than a domain-specific tool-use simulator.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not include **stateful tool-use** and **API lifecycle** aspects, it may miss an important class of long-horizon failures in deployed assistants.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short related-work paragraph distinguishing “stateful tool-use lifecycle benchmarks” (DialogTool/VirtualMobile) from GALILEO’s targeted phenomenon, and cite that long-horizon failures appear in tool-use too.
- [ ] Consider whether GALILEO should include a small “tool state” condition (even toy tools) to test whether GALILEO-style drift/pressure metrics correlate with tool-use degradation.

## Quotes / details to potentially cite

- “Existing benchmarks … primarily focus on stateless, single-turn interactions or partial evaluations … overlooking the inherent stateful nature of interactions in multi-turn applications.”
- “We propose DialogTool … considering the whole life cycle of tool use … across six key tasks in three stages …”
- “We build VirtualMobile … to simulate API calls and assess the robustness of the created APIs …”
- “… existing state-of-the-art LLMs still cannot perform well to use tools over long horizons.”
