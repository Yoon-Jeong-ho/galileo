# Ask-to-Clarify: Resolving Instruction Ambiguity through Multi-turn Dialogue

- Year: 2025
- Venue: arXiv
- Authors: Xingyao Lin; Xinghao Zhu; Tianyi Lu; Sicheng Xie; Hui Zhang; Xipeng Qiu; Zuxuan Wu; Yu-Gang Jiang
- URL: https://arxiv.org/abs/2509.15061
- BibTeX key (if we add it): ask_to_clarify_2025
- Tags: embodied agents, VLA, clarification questions, multi-turn dialogue, diffusion policy

## One-sentence takeaway

A two-component VLA (VLM dialogue + diffusion action expert) that explicitly asks clarifying questions for ambiguous instructions, then executes via low-level actions, improving real-world task success.

## What problem does it solve?

- Real-world robot instructions are frequently ambiguous (e.g., multiple candidate objects); most VLAs execute one-shot without querying the user, leading to random choices or failure.
- Prior “ask” approaches often live in simulation and/or rely on high-level action abstractions; this work targets real-world, low-level control.

## What is the core method / protocol?

- **Architecture:**
  - **Collaboration component:** a VLM that outputs (i) clarifying questions in dialogue form and (ii) signals about whether to ask vs act.
  - **Action component:** a **diffusion** policy/expert for low-level robot actions.
  - **Connection module:** produces the conditioning for the diffusion model from the VLM output; described as adjusting the observation using the instruction to create more reliable conditions.
- **Two-stage “knowledge-insulation” training:**
  1) Fine-tune the VLM on ambiguity-solving dialogue data to learn when/how to ask.
  2) Freeze the collaboration VLM; integrate and fine-tune the diffusion action expert for low-level action generation, preserving the learned interaction behavior.
- **Inference routing:** a signal detector/router switches between “ask a question” mode and “take actions” mode.

## What are the key metrics?

- Real-world task success across **8 tasks** (paper claims significant gains vs SOTA VLAs).
- (From the abstract/intro) comparative evaluation vs π0, π0-FAST, OpenVLA-OFT; details not captured in this rapid skim.

## What are the main results?

- Reported to **significantly outperform** existing state-of-the-art VLAs on 8 real-world tasks where ambiguous instructions require interaction.
- Key claim: the two-stage training prevents losing “ask” capability when training the low-level action module.

## How is this similar to GALILEO?

- Frames embodied agents as **collaborators** with humans rather than one-way executors.
- Uses explicit **interaction / query** to resolve underspecified goals before committing to actions.
- Emphasizes real-world robustness beyond purely simulated dialogue datasets.

## How is this different from GALILEO?

- Very specific **ask-then-act** pipeline: a dialogue VLM + diffusion policy with a router; GALILEO may (depending on our framing) aim for broader goal inference / planning / grounding.
- The “connection module” suggests a particular conditioning interface between language reasoning and low-level diffusion actions.
- Primary focus appears to be ambiguity resolution via question-asking, not (e.g.) long-horizon task decomposition or environment/model-based planning.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a more principled uncertainty model or information-gain-based querying, that could be a cleaner justification than a supervised router token.
- If GALILEO unifies asking and acting in a single policy (or provides guarantees), that could be simpler than multi-component integration.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not yet have strong, benchmarked **multi-turn clarification** behavior, this paper is a direct competitive reference point.
- If GALILEO uses high-level actions only, this work highlights the importance of **low-level** end-to-end control for “real” manipulation.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a related-work paragraph contrasting “ask-to-clarify” frameworks vs one-shot VLAs (and vs simulated ask-to-act style work).
- [ ] Consider an ablation idea: freeze vs not-freeze the language/dialogue component when training a low-level action module (catastrophic forgetting of interaction).
- [ ] If applicable, implement a simple “router” baseline (ask vs act) to compare with any unified GALILEO approach.

## Quotes / details to potentially cite

- “Our framework first resolves ambiguous instructions by asking clarifying questions in a multi-turn dialogue. Then it generates low-level actions … end-to-end.” (abstract)
- “We train our framework with a two-stage knowledge-insulation training strategy … fine-tune the collaboration component … then … freeze … while fine-tuning the diffusion expert to generate low-level actions.” (abstract)
- Evaluated “in 8 real-world tasks” and compared to “π0, π0-FAST and OpenVLA-OFT.” (intro/abstract)
