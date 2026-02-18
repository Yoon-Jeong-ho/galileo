# On the Failure of Latent State Persistence in Large Language Models

- Year: 2025
- Venue: arXiv (cs.CL, cs.AI) — notes mention “Machine Learning, ICML” (likely submission)
- Authors: Jen-tse Huang; Kaiser Sun; Wenxuan Wang; Mark Dredze
- URL: https://arxiv.org/abs/2505.10571
- BibTeX key (if we add it): huang2025failure-lsp (suggested)
- Tags: multi-turn, latent-state, persistence, working-memory, consistency, evaluation

## One-sentence takeaway

Across three interactive “hidden state” games, frontier LLMs behave like post-hoc constraint solvers rather than agents with a persistent internal variable, producing measurable probability-mass and consistency violations when the state is not explicitly written in context.

## What problem does it solve?

- How to *operationalize and test* whether an LLM can maintain a private latent variable (a working-memory-like state) across turns when that variable is never externalized in the prompt/context.
- Provides concrete, falsifiable indicators for “latent state persistence” (LSP) instead of conflating working memory with long-context retrieval.

## What is the core method / protocol?

- Defines **Latent State Persistence (LSP)** as the ability to instantiate/maintain/manipulate an internal variable \(x\) across multiple interactions without writing it into the context.
- Proposes three experiments that derive **testable mathematical consequences** of LSP:
  1) **Number Guessing Game**: model is instructed to “think of” an integer \(x\in\{1..n\}\) privately; then queried independently “Is it i? Yes/No.”
     - With true persistent unique \(x\), the probabilities must satisfy a **Sum-of-Probability Identity**: \(\sum_i P(\text{Yes}|Q_i)=1\).
     - Defines **Empirical State Mass (ESM)** as \(\sum_i \hat P(\text{Yes}|\mathcal T_i)\) from repeated sampling.
  2) **Yes–No Game** (20-questions-like) over an object space with ordered attributes; queries ask comparisons vs reference objects.
     - With persistent \(x\), each answer is O(1) (just compare with \(x\)); without persistence, model must solve a growing constraint system and tends to “concept drift” into contradictions.
     - Uses **Mean Steps to Contradiction** and **Pass Rate** (survive up to T=250 queries without contradiction).
  3) **Mathematical Mentalism**: track deterministic transformations on hidden variables; tests variable binding and state evolution when initial state is not present.

- Evaluates **17 LLMs** (GPT-4o/mini, OpenAI o-series, LLaMA, Qwen, DeepSeek, etc.) under different decoding settings; runs repeated trials for empirical probabilities.

## What are the key metrics?

- **ESM (Empirical State Mass)** in Number Guessing Game; deviation from the theoretical identity ESM=1.
- **Steps-to-contradiction / MSC** (mean steps to contradiction) and **PR** (pass rate to horizon) in the Yes–No game.
- (Qualitative) systematic biases like the “blue-seven” heuristic in number selection/affirmation.

## What are the main results?

- **Mass under-allocation (ESM→0)** is common: models often answer “No” to nearly all i, consistent with treating each query as a frequentist event with prior 1/n rather than consulting a committed hidden \(x\).
- When “Yes” is produced, it is frequently **non-uniform and biased** (notably toward **7** / “blue-seven” effects), inconsistent with a stable sampled \(x\).
- In the Yes–No game, models exhibit **concept drift / self-contradiction** as constraints accumulate; better models last longer, but failures persist.
  - The paper reports, e.g., **GPT-4o-Mini PR=0%** and GPT-4o having a **non-zero PR (~13.5%)** at T=250 in their setup—interpreted as stronger constraint-handling, not evidence of true LSP.
- “Long reasoning traces”/extra compute can help by **externalizing** intermediate reasoning, but does not close the underlying LSP gap when the hidden state is absent from context.

## How is this similar to GALILEO?

- If GALILEO is targeting robust multi-turn reasoning and agent-like behavior, this paper is directly aligned: it isolates a core failure mode of typical LLM agents—**lack of stable hidden state across turns**—that impacts planning, consistency, and tool-use loops.
- The evaluation style (interactive protocols + clear invariants) is a good template for GALILEO-facing benchmarks.

## How is this different from GALILEO?

- This work is primarily **diagnostic/benchmarking + theory framing**; it does not propose an architectural fix beyond pointing to the need for explicit state grounding/externalization.
- Focuses on *pure language interaction* without tool-augmented external memories/state stores (which GALILEO may leverage).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO maintains an explicit state object / scratchpad / external memory, it can *avoid* the “hidden mind” ambiguity by designing protocols where the state is tracked in a controlled channel/store.
- GALILEO can demonstrate reliability by coupling actions to a verified state representation rather than relying on implicit latent persistence.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently assumes the model “remembers” a hidden choice without writing it down, this paper suggests that assumption is unsafe; you likely need explicit state externalization (even if compressed).
- If GALILEO claims internal working-memory-like persistence, you will need evidence under protocols where the state is *never* present in context.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an **LSP evaluation section** (or appendix) modeled on these three games; report invariants/metrics (ESM; contradiction horizon).
- [ ] Run ablations: (a) hidden state only (no externalization), (b) explicit state written each turn, (c) state stored in an external memory/tool and retrieved.
- [ ] If using tool memory, define and measure **state fidelity** (does the agent consult/update the correct state, and does it remain consistent over long horizons?).
- [ ] Consider incorporating an “ESM-like” probabilistic completeness check for any GALILEO component that claims to sample/commit to a discrete hidden variable.

## Quotes / details to potentially cite

- Formal identity for persistent unique hidden state (Number Guessing): “the sum of the probabilities of answering ‘Yes’ across all possible queries … must equal unity.” (Proposition 2.1; Sum-of-Probability Identity)
- Framing: lack of LSP implies LLMs act as “reactive post-hoc solvers rather than proactive planners with LSP.”
- Observed heuristic: strong bias toward 7 (“blue-seven” phenomenon) in affirmative responses/choices.
