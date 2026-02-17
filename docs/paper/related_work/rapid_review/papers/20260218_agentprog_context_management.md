# AgentProg: Empowering Long-Horizon GUI Agents with Program-Guided Context Management

- Year: 2025
- Venue: arXiv preprint
- Authors: Hao Wen; Yuxuan Chen; Jiacheng Liu; Shanhui Zhao; Guohong Liu; Ju Ren; Yunxin Liu; Yuanchun Li
- URL: https://arxiv.org/abs/2512.10371
- BibTeX key (if we add it): agentprog_wen_2025
- Tags: gui-agents, long-horizon, context-management, memory, planning, program-guided, androidworld

## One-sentence takeaway

Representing an agent’s long interaction history as an executable-ish “semantic task program” with variables/control flow (plus a global belief state) provides a principled way to retain task-critical information and stay robust on long-horizon mobile tasks.

## What problem does it solve?

- Long-horizon mobile GUI agents accumulate ever-growing interaction histories, causing (a) context-window / cost blow-up and (b) forgetting or losing task-critical state under naïve truncation/summarization.
- Existing context management (sliding windows, per-step summaries, hierarchical planning) is not task-aware enough to decide what to keep vs discard, leading to sharp performance degradation as horizons grow.

## What is the core method / protocol?

- Reframe the interaction history as a **program**:
  - Introduce **Semantic Task Program (STP)**: a domain-specific representation mixing natural-language-like instructions with **structured control flow** (loops/branches/functions) and **explicit variables**.
  - Key idea: task decomposition + memory are represented by program structure (control flow + data flow), giving a *principled retention rule*: keep variable values / relevant frames for the currently-active parts of the program; discard irrelevant navigation details.
- Add a **Global Belief State** inspired by Belief MDPs to handle partial observability and dynamic GUI state:
  - Maintain/validate/update hypotheses about environment state (including hidden/implicit variables), so execution can adapt when the UI/environment changes or earlier assumptions become stale.
- Evaluation introduced **AW-Extend** (built on AndroidWorld): 19 long-horizon tasks, including compositional tasks and iterative tasks scaling subtasks (e.g., n=10/20) to stress memory filtering and robustness.

## What are the key metrics?

- Task success rate on **AndroidWorld** and **AW-Extend** (long-horizon suite).
- Robustness vs horizon length (whether performance degrades “catastrophically” as tasks get longer).

## What are the main results?

- Claims new SOTA success rates on AndroidWorld and strong gains on AW-Extend.
- Key qualitative result: baselines that do well on AndroidWorld can **collapse on AW-Extend**, while AgentProg maintains relatively stable performance as horizon/subtask count increases.

## How is this similar to GALILEO?

- Same overall motivation: **long-horizon agent behavior needs structured state/memory**, not an ever-growing raw transcript.
- Both implicitly argue that *task structure* should drive context selection and that robustness hinges on maintaining the right latent/task state across many steps.

## How is this different from GALILEO?

- Their core lens is a **program analogy** (STP with variables/control flow) for context management; GALILEO may use different representations (e.g., latent state, retrieval, hierarchical memory) depending on our design.
- They explicitly add a **global belief state** module (Belief-MDP-inspired) to handle partial observability and environment changes.
- Domain focus: **mobile GUI agents** (AndroidWorld) and their custom AW-Extend suite.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s memory/state mechanism is model-agnostic and simpler to integrate, it may avoid the overhead/complexity of generating/maintaining an STP.
- If GALILEO supports more general environments beyond mobile GUIs, it may have broader claims.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks an explicit representation of **control flow + data flow** (loops/variables) it may struggle on iterative/compositional long tasks where “what to remember” is naturally variable-centric.
- If GALILEO doesn’t maintain an explicit belief state for hidden/implicit environment variables, robustness to partial observability may be weaker.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding/ablation: **control-flow-aware context pruning** (e.g., keep state tied to active subtask/loop iteration) vs generic summarization.
- [ ] Add/ablation: **belief-state consistency checks** for partially observable GUI-like settings (even if simulated), to test robustness under hidden state shifts.
- [ ] In related-work writing: position STP as “program-guided context management” and cite as evidence that *program structure* can serve as an effective memory policy for long horizons.

## Quotes / details to potentially cite

- “reframes the interaction history as a program with variables and control flow” (abstract)
- Introduces STP: “fuzzy, natural-language-style instructions” + structured control flow; interpreted adaptively at runtime based on environment state (intro).
- AW-Extend: extended suite to stress compositional + iterative long-horizon tasks; baselines can show “catastrophic degradation” while their method stays robust (motivation/abstract).
