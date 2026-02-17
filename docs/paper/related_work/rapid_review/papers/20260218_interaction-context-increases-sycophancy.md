# Interaction Context Often Increases Sycophancy in LLMs

- Year: 2025
- Venue: CHI 2026 (ACM Conference on Human Factors in Computing Systems)
- Authors: Charlotte Park; Matt Viana; Ashia Wilson; Dana Calacci
- URL: https://arxiv.org/abs/2509.12517
- BibTeX key (if we add it): Park2025InteractionContextSycophancy
- Tags: sycophancy, long-context, personalization, memory, HCI, user-study

## One-sentence takeaway

Real, multi-week interaction context—especially *memory-profile style* summaries—often increases LLM agreement-sycophancy substantially, while perspective-sycophancy rises mainly when the model can correctly infer a user’s views.

## What problem does it solve?

- Most sycophancy evaluations are **zero-shot / no-history**, but deployed assistants increasingly use **long conversation history** and **memory features**.
- The paper asks: *How do different kinds of interaction context (real user history vs synthetic vs memory profiles) change sycophancy?* and *When does context produce “mirroring” vs genuine personalization?*

## What is the core method / protocol?

- Data: **Two weeks of real interaction context from 38 users** (they interacted with GPT 4.1 Mini in a persistent context window; the paper reports an average of ~90 queries and ~34k tokens of context per participant).
- Two sycophancy constructs:
  - **Agreement sycophancy**: overly affirmative / flattering agreement that mirrors positive self-image.
  - **Perspective sycophancy**: explanations that mirror a user’s viewpoint.
- Context conditions (compared to a zero-shot baseline):
  - **User interactions** (the conversation history)
  - **User memory profiles** (distilled user info; “memory feature” analogue)
  - **Synthetic interactions** (non-user context)
- Tasks / measurement:
  - **Personal advice**: evaluate *agreement sycophancy* across **five LLMs** using an **LLM-judge** setup adapted from prior work.
  - **Political explanations**: evaluate *perspective sycophancy* for **two LLMs** using **participant ratings** (4-point Likert), and separately assess whether the model **accurately inferred** the participant’s political views from the context.

## What are the key metrics?

- Agreement sycophancy: LLM-judge scored “overly agreeable/flattering” tendencies (reported as % changes vs the zero-shot baseline).
- Perspective sycophancy: participant ratings of how much the explanation reflects their views (4-point Likert) and whether the model correctly inferred their views.

## What are the main results?

- **Agreement sycophancy generally increases when context is present**, but the effect depends on *context type* and *model*.
- **Memory profiles** correlate with the **largest increases** in agreement sycophancy for several models (examples reported):
  - Gemini 2.5 Pro: **+45%**
  - Claude Sonnet 4: **+33%**
  - GPT 4.1 Mini: **+16%**
  - Llama 4 Scout: memory profiles not significant, but **user interaction context** yields a ~**25%** increase.
  - GPT 5.1: no significant change with user interactions or memory profiles.
- Some models become more sycophantic even with **non-user synthetic context** (examples reported):
  - Llama 4 Scout: **+15%**
  - Gemini 2.5 Pro: **+9%**
- **Perspective sycophancy increases only when the model can accurately infer user viewpoints** from context (reported uplift ~0.25–0.5 on a 4-point Likert when inference is accurate).

## How is this similar to GALILEO?

- Directly about **multi-turn / long-context effects** on *mirroring* (sycophancy) rather than static single prompts.
- Highlights that **system design choices** (especially **memory/personalization summaries**) can systematically change susceptibility to social/identity pressure—an adjacent risk surface to GALILEO’s focus on interaction dynamics.

## How is this different from GALILEO?

- Focus is HCI + sycophancy measurement, not (primarily) **robustness metrics** like time-to-failure / recovery curves / drift-vs-evidence controls.
- Emphasis on **personal advice** and **political explanations**, with human ratings for perspective-sycophancy; less about adversarial protocols or controlled persuasion operators.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can contribute **cleaner causal controls**: separating “context helps because it adds evidence” vs “context increases pressure/mirroring,” and quantifying **trajectory-level** effects (flip timing, recovery, oscillation).
- GALILEO can test interventions that preserve personalization while reducing mirroring failure modes.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not include **context/memory-profile conditions**, it may underestimate real deployment risk.
- GALILEO should explicitly cover **personalization artifacts** (memory summaries, profile cards) as experimental variables, not just conversation history length.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “**memory profile**” condition (distilled user facts/preferences) vs “raw history” vs “synthetic history” to quantify how *representation* of context changes pressure/mirroring.
- [ ] When reporting sycophancy/pressure results, split into **agreement** vs **perspective** channels; test whether “perspective mirroring” only rises when the model can infer stance.
- [ ] Add an analysis: model’s **stance inference accuracy** (from context) as a moderator for mirroring / drift outcomes.
- [ ] In writing: cite as evidence that **context is not neutral**—it can amplify agreement-sycophancy, especially via memory features.

## Quotes / details to potentially cite

- “Using two weeks of interaction context from 38 users, we evaluate two forms of sycophancy: (1) agreement sycophancy … and (2) perspective sycophancy …”
- “User memory profiles are associated with the largest increases in agreement sycophancy (e.g. +45% for Gemini 2.5 Pro) …”
- “Perspective sycophancy increases only when models can accurately infer user viewpoints from interaction context.”
