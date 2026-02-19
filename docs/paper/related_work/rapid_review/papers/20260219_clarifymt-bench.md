# ClarifyMT-Bench: Benchmarking and Improving Multi-Turn Clarification for Conversational Large Language Models

- Year: 2025
- Venue: arXiv
- Authors: Sichun Luo, Yi Huang, Mukai Li, Shichang Meng, Fengyuan Liu, Zefa Hu, Junlan Feng, Qi Liu
- URL: https://arxiv.org/abs/2512.21120
- BibTeX key (if we add it): clarifymtbench2025luo
- Tags: multi-turn, clarification, ambiguity, benchmark, user-simulation

## One-sentence takeaway

ClarifyMT-Bench shows that even strong LLMs systematically **under-clarify** (answer too early) in noisy multi-turn ambiguity settings, and proposes an agentic decomposition (ClarifyAgent) that improves clarify-vs-answer robustness.

## What problem does it solve?

- Existing clarification benchmarks tend to be **single-turn**, assume **cooperative users**, or are **task/domain-limited**, so they miss real conversational failure modes: vague/contradictory/off-focus/factually-wrong user replies across multiple turns.
- The paper targets the decision policy: **when should a model ask a follow-up vs. answer now**, as interaction depth increases.

## What is the core method / protocol?

- **Benchmark:** ClarifyMT-Bench with 6,120 multi-turn dialogues.
  - Grounded in a **five-dimensional ambiguity taxonomy**:
    - linguistic, intent, contextual, epistemic, interactional ambiguity.
  - Uses **six simulated user personas/behaviors** to inject realistic noise (e.g., vague, contradictory, off-focus, factually wrong replies).
- **Evaluation task:** per turn, predict action in {Clarify, Answer} (decision accuracy; analyze under- vs over-clarification).
- **Method baseline:** ClarifyAgent, an agentic pipeline decomposing the decision into modules:
  - Perceiver (extract info + detect ambiguity)
  - Forecaster (infer user persona / behavioral tendency)
  - Tracker (maintain unresolved “slots” / conflicts)
  - Planner (choose clarify vs answer)
  - Output module executes the chosen action

## What are the key metrics?

- Turn-level **decision accuracy** for Clarify vs Answer.
- Error modes:
  - **Under-clarify:** answers when reference action is Clarify.
  - **Over-clarify:** clarifies when reference action is Answer.
- Robustness slices called out in the paper’s framing:
  - performance vs **dialogue depth**
  - performance vs **ambiguity type**
  - performance vs **user persona/behavior**

## What are the main results?

- Across 10 representative LLMs, the authors report a consistent **under-clarification bias** (premature answering).
- Performance **degrades as dialogue depth increases**, suggesting brittle state tracking / uncertainty handling over multi-turn contexts.
- ClarifyAgent improves robustness across ambiguity conditions (reported as “substantial improvements”; exact deltas not captured in this rapid pass).

## How is this similar to GALILEO?

- Shares the broader theme of **multi-turn robustness under interactional noise** (the user can be unhelpful, inconsistent, or adversarial-ish without being explicitly an attacker).
- Emphasizes **trajectory effects** (depth/horizon matters), not just single-shot behavior.

## How is this different from GALILEO?

- Focuses on **clarification policy** (ask vs answer) for open-domain assistants, not GALILEO’s core setting/tasks.
- Uses simulated personas + taxonomy-driven ambiguity generation rather than GALILEO-style environments/datasets.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO explicitly separates *evidence-driven revision* vs *social-pressure drift* (or has stronger causal controls), that would be cleaner than a generic clarify/answer benchmark.
- If GALILEO has stronger longitudinal metrics (recovery, hazards, survival-style reporting), it can go beyond simple turn-level decision accuracy.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently lacks an explicit **ask-vs-answer / abstain / query-for-info** action space, ClarifyMT-Bench highlights a missing capability that matters for safe/robust interaction.
- GALILEO might benefit from an explicit taxonomy of **ambiguity sources** (not only “attack types”).

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a small “clarify-or-answer” auxiliary task or evaluation slice (even synthetic) to test whether GALILEO-trained systems avoid premature confident answers under underspecification.
- [ ] Add an “interactional ambiguity” category in related work (vague/contradictory/off-focus user turns) and cite ClarifyMT-Bench as evidence of under-clarification bias.
- [ ] If we build agentic policies, consider module decomposition analogous to **perception / forecasting / tracking / planning**.

## Quotes / details to potentially cite

- ClarifyMT-Bench: “a benchmark for multi-turn clarification grounded in a five-dimensional ambiguity taxonomy and a set of six behaviorally diverse simulated user personas.”
- Empirical claim: “a consistent under-clarification bias: LLMs tend to answer prematurely, and performance degrades as dialogue depth increases.”
- ClarifyAgent: “decomposes clarification into perception, forecasting, tracking, and planning.”
