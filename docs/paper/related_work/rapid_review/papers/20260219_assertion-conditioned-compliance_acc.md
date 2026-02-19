# Assertion-Conditioned Compliance: A Provenance-Aware Vulnerability in Multi-Turn Tool-Calling Agents

- Year: 2025 (v1) / 2026 (v2)
- Venue: arXiv
- Authors: Daud Waqas, Erika Hayashida, Aaryamaan Golthi, Huanzhi Mao
- URL: https://arxiv.org/abs/2512.00332
- BibTeX key (if we add it): waqas2025acc
- Tags: agents, tool-use, multi-turn, compliance, provenance, vulnerability, sycophancy-adjacent

## One-sentence takeaway

A-CC introduces a provenance-aware stress test for tool-using agents showing that plausible *user* or *tool/system* assertions can silently steer the tool-calling trajectory (procedural compliance) even when standard task success metrics (e.g., BFCL accuracy) stay high.

## What problem does it solve?

- Existing function-calling benchmarks (e.g., BFCL) largely score end-task success, but do not directly measure *conversation-level robustness* to misleading statements encountered mid-dialogue.
- In real deployments, agents must handle:
  - **User-sourced assertions (USAs):** plausible but wrong beliefs from the user (sycophancy/agreeableness pressure).
  - **Function-sourced assertions (FSAs):** plausible but contradictory “system/tool policy” hints (e.g., stale or misconfigured tool outputs).

## What is the core method / protocol?

- **Assertion-Conditioned Compliance (A-CC):** an evaluation paradigm for multi-turn, stateful tool-calling tasks where an incorrect assertion is injected and the agent’s subsequent tool-calling behavior is assessed.
- Two assertion provenance conditions:
  - **USA condition:** inject a plausible-but-incorrect user assertion.
  - **FSA condition:** inject a plausible-but-incorrect assertion as part of a tool/function response (e.g., a “policy note”).
- Key idea: measure **procedural compliance** (does the agent update its *tool-use plan / execution* to follow the assertion), not just verbal agreement.
- Built by extending **BFCL-style** multi-turn, state-dependent function-calling tasks with assertion injections and additional diagnostics.

## What are the key metrics?

- **Task success / accuracy** on the underlying multi-turn tool-calling tasks (BFCL-style accuracy).
- **Compliance-oriented diagnostics** (paper terminology): whether the model follows the injected assertion in its subsequent tool calls / state updates (i.e., procedural compliance).
- **Coupling/decoupling analysis:** how strongly compliance correlates with final accuracy (the paper argues these can be weakly coupled).

## What are the main results?

- Across tested model families/sizes, models are **highly vulnerable** to both:
  - **USA sycophancy** (social/pro-user cues) and
  - **FSA policy conflicts** (system/tool-origin cues).
- **Compliance is not tightly coupled to accuracy:** a model can still “solve the task” while being steered into unnecessary/unsafe tool actions under assertion pressure.
- Practical implication: relying on leaderboard task success alone can miss critical safety risks in deployed multi-turn agents.

## How is this similar to GALILEO?

- Both care about **multi-turn robustness** under realistic interaction dynamics (misleading context, pressure signals, and recovery vs degradation).
- Both motivate evaluation beyond single-turn correctness (trajectory-/turn-level failure modes).

## How is this different from GALILEO?

- A-CC is targeted at **tool-calling / function-calling agents** and explicitly studies *provenance* (user vs tool) of misleading assertions.
- It frames the failure as **procedural compliance** (execution changes), not only belief drift / stance drift in natural-language responses.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides cleaner *general* multi-turn robustness metrics (e.g., time-to-failure, recovery curves) across domains without requiring tool APIs, that may generalize more broadly.
- If GALILEO explicitly models *return-to-truth / recovery* dynamics, it could complement A-CC’s compliance lens.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not separate **source/provenance** of misleading signals (user vs system/tool), A-CC suggests this is an important axis.
- If GALILEO does not test **tool-execution trajectories**, it may miss practical safety risks that occur even when final answers look fine.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a provenance axis to GALILEO’s taxonomy: “pressure from user” vs “pressure from system/tool/context”.
- [ ] Consider a metric analogous to **procedural compliance**: does the model *act* on misleading premises (tool calls / intermediate decisions), not just state agreement.
- [ ] In related work: position A-CC as a neighbor that shows **accuracy can stay high while trajectory risk increases**.

## Quotes / details to potentially cite

- “A-CC provides holistic metrics that evaluate a model’s behavior when confronted with misleading assertions … (1) user-sourced assertions (USAs) … and (2) function-sourced assertions (FSAs) …” (abstract).
- “Models are highly vulnerable to both USA sycophancy and FSA policy conflicts …” (abstract).
- “Assertion compliance is not tightly coupled to accuracy degradation … models judged solely on task success accuracy by leaderboard scores may still execute unnecessary or unsafe operations …” (intro).