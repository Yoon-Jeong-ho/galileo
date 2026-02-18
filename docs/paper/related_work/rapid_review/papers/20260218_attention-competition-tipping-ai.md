# Competition for attention predicts good-to-bad tipping in AI

- Year: 2026
- Venue: arXiv
- Authors: Neil F. Johnson
- URL: https://arxiv.org/abs/2602.14370
- BibTeX key (if we add it): johnson2026attention_tipping
- Tags: dynamics, multi-step, conversations, tipping, attention

## One-sentence takeaway

The paper proposes a mechanistic, attention-based dynamical model that predicts when a multi-turn conversation will “tip” from good to bad outputs, yielding a closed-form tipping point n* driven by dot-product competition between the context vector and competing output basins.

## What problem does it solve?

- Predicting *when* (at which turn/step) an LLM in deployment will transition from benign/helpful outputs to harmful/undesirable ones, especially in offline/on-device (“edge”) settings where cloud guardrails, monitoring, and patching are unavailable.
- Framing safety failures as *dynamical tipping* phenomena rather than purely static “is this prompt unsafe?” filtering.

## What is the core method / protocol?

- Coarse-grain model outputs into symbolic “basins” (e.g., A=neutral, B=good/desirable, D=bad/undesirable, C=other) for a given topic/task.
- Treat the next-step generation as arising from *competition for attention*:
  - The conversation history produces a context vector c (from attention-weighted combination of prior tokens/symbols).
  - Candidate basins compete via dot products c·B vs c·D (and implicitly softmax), especially transparent at low decoding temperature.
- Derive a mathematical expression for the tipping point n* (the step at which D starts winning), governed by how additional conversational content aligns the context toward/away from the undesirable basin.
- Validate qualitatively/quantitatively across multiple transformer LMs (GPT-2 variants and several small open models; plus an external check on a production model).

## What are the key metrics?

- Tipping point n* (turn index / step count when the output transitions into the undesirable basin).
- Frequency/rate of good→bad vs bad→good transitions across conversation sequences (as induced by history).
- Agreement of predicted tipping behavior with observed symbol sequences under controlled prompts (often at low temperature to reduce stochasticity).

## What are the main results?

- Good-to-bad tipping can happen immediately or after a long benign run, and is strongly history-dependent in multi-step conversations.
- A single additional piece of content (their “C”) can steer the model toward or away from tipping by changing alignment of the context vector relative to competing basins.
- The proposed attention-competition mechanism yields a predictive handle (n*) and suggests “control levers” (design conversation/context so c is less aligned with D).
- Empirical demonstrations on multiple LMs suggest the phenomenon is architecture-agnostic within the tested transformer family.

## How is this similar to GALILEO?

- Both are concerned with multi-turn conversational dynamics and state/history dependence (not just single-turn prompt safety).
- Both motivate *predictive* signals for trajectory shifts (e.g., early warning of drift/tipping) and potential interventions that steer future responses.

## How is this different from GALILEO?

- This paper is primarily a mechanistic/dynamical-theory framing centered on attention dot-product geometry and coarse-grained basins; it does not propose a full end-to-end system for conversational control, monitoring, or optimization beyond the tipping analysis.
- It focuses on “good vs bad” content categories; it is less about task success, conversational goals, or structured grounding (depending on how GALILEO defines its objectives).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides explicit objectives, control policies, or evaluation suites for multi-turn behavior, it can be positioned as a more actionable framework than a descriptive tipping theory.
- If GALILEO uses richer state representations than coarse symbol basins, it may capture more nuance than A/B/C/D coarse-graining.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a simple, mechanistically interpretable scalar indicator like n* (or an attention-competition proxy), this work is a useful contrast: it offers an interpretable “why/when tipping happens” story that can strengthen GALILEO’s related-work narrative.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related-work: cite this as a “dynamical tipping / attention competition” mechanism for multi-turn safety failures, especially for offline/on-device LMs.
- [ ] Add an analysis experiment: compute an attention-based proxy (context alignment to “undesirable” directions) and test whether it predicts future drift/tipping in GALILEO’s setting.
- [ ] Consider a controllability angle: can GALILEO interventions be described as pushing the context vector away from undesirable basins?

## Quotes / details to potentially cite

- Abstract-level framing: safety tooling often assumes cloud connectivity; offline/on-device LMs lack monitoring/patching.
- Key technical claim: tipping point n* is governed by dot-product competition for attention between the conversation’s context and competing output basins (good vs bad).
- Demonstration claim: multi-step conversations can flip answers to high-stakes questions depending on earlier, seemingly unrelated context.
