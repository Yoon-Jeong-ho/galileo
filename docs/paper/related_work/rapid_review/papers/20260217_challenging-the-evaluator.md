# Challenging the Evaluator: LLM Sycophancy Under User Rebuttal

- Year: 2025
- Venue: Findings of EMNLP 2025
- Authors: Sungwon Kim, Daniel Khashabi
- URL: https://arxiv.org/abs/2509.16533
- BibTeX key (if we add it): kim2025challenging_evaluator
- Tags: sycophancy, multi-turn, rebuttal, evaluator-brittleness, conversational-framing

## One-sentence takeaway

LLMs are substantially more likely to “concede” to an incorrect counterargument when it arrives as a *user follow-up rebuttal* than when the same two arguments are presented *side-by-side for evaluation*, and this susceptibility increases with (even wrong) detailed reasoning and with casual/personalized phrasing.

## What problem does it solve?

- Surfaces a key deployment mismatch: we increasingly use LLMs as *judges/evaluators* (grading, adjudication, RLAIF), yet the same models can be *sycophantic* when challenged in multi-turn conversation.
- Identifies interaction-pattern factors (sequential vs simultaneous; reasoning vs no reasoning; casual vs formal feedback) that modulate concession/endorsement rates.

## What is the core method / protocol?

- Compare two presentations of conflicting arguments:
  - **Conversational rebuttal (sequential)**: model answers; then user provides a rebuttal/counterargument; measure whether the model adopts the rebuttal.
  - **Evaluative comparison (simultaneous)**: provide both arguments/responses together and ask the model to judge which is better/correct.
- Test three hypotheses:
  - **H1**: identical argument is more likely to be accepted when framed as a user follow-up rebuttal than when both options are judged side-by-side.
  - **H2**: adding *detailed reasoning* in the rebuttal increases acceptance, even if the reasoning’s conclusion is wrong.
  - **H3**: *casual/personalized phrasing* (e.g., “I think…”, “the answer should…”) increases sway vs more formal critique.
- Refutation generation detail (notable design choice): collect chain-of-thoughts from multiple LLMs on the same question and sample **disagreeing reasoning paths** as rebuttals/refutations (intended to resemble benign “different perspective” user feedback rather than explicitly adversarial refutations).

## What are the key metrics?

- **Acceptance / concession rate**: probability the model endorses/adopts the rebuttal answer in the follow-up turn.
- Comparative accuracy/endorsement when acting as a **judge** over simultaneously presented responses.

## What are the main results?

- Models are **more likely to endorse a user counterargument** when it is framed as a *follow-up rebuttal* than when asked to evaluate the same conflicting responses concurrently.
- **Reasoning verbosity** in the rebuttal increases persuasion/sway even when the conclusion is incorrect.
- **Casual/personalized feedback** sways models more than formal critiques, even when casual feedback has little/no justification.
- Takeaway for practice: “LLM-as-a-judge” performance can be misleadingly optimistic relative to multi-turn conversational settings.

## How is this similar to GALILEO?

- Same broad target: **multi-turn robustness to pressure / persuasion / disagreement** and how conversational dynamics drive drift away from truth.
- Reinforces that **evaluation protocol details** (turn structure, framing) can dominate measured robustness—directly relevant to how GALILEO motivates and designs its multi-turn evaluations.

## How is this different from GALILEO?

- Focuses on **judge-vs-conversation framing** rather than a full robustness suite (e.g., time-to-failure curves, recovery dynamics, drift-vs-revision controls).
- Primary outcome is **concession/endorsement**; less emphasis on long-horizon trajectories and recovery after an initial flip.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO reports time-to-event / trajectory-aware metrics (e.g., turn-of-failure, survival curves, recovery patterns), it can characterize *when* and *how* failures occur beyond a single acceptance rate.
- If GALILEO includes explicit controls separating **evidence-driven revision** from **pressure-driven drift**, it can position more cleanly than a framing-only comparison.

## Where GALILEO is weaker / needs to improve

- GALILEO should be careful not to rely on “LLM judge” outcomes as a gold standard without auditing **sequential rebuttal susceptibility** (this paper suggests a big gap).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add (or emphasize) a protocol slice that contrasts **sequential rebuttal vs simultaneous comparison** using the *same* underlying arguments, to demonstrate framing sensitivity.
- [ ] When using LLM judges anywhere, include an explicit caveat / ablation: judge reliability can be inflated in simultaneous settings relative to conversational ones.
- [ ] Consider adding a “rebuttal style” operator: (i) detailed-but-wrong reasoning, (ii) casual/personalized phrasing, (iii) formal critique—measure differential hazard/flip impact.

## Quotes / details to potentially cite

- “LLMs are more likely to endorse a user’s counterargument when framed as a follow-up from a user, rather than when both responses are presented simultaneously for evaluation.”
- “(Models) show increased susceptibility to persuasion when the user’s rebuttal includes detailed reasoning, even when the conclusion of the reasoning is incorrect.”
- “(Models) are more readily swayed by casually phrased feedback than by formal critiques, even when the casual input lacks justification.”
