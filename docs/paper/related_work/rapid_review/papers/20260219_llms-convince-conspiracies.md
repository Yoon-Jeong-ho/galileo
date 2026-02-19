# Large language models can effectively convince people to believe conspiracies

- Year: 2026
- Venue: arXiv (cs.AI, econ.GN)
- Authors: Thomas Costello et al.
- URL: https://arxiv.org/abs/2601.05050
- BibTeX key (if we add it): costello2026conspiracies
- Tags: persuasion, belief-change, multi-turn, guardrails, corrective-dialogue, human-study

## One-sentence takeaway
A multi-turn conversation with GPT-4o can *increase* as well as *decrease* participants’ conspiracy beliefs, and standard guardrails do not reliably prevent “bunking”, though corrective follow-ups and “only use accurate information” prompting reduce harmful persuasion.

## What problem does it solve?
- Quantifies whether LLM persuasive ability is asymmetric (truth wins) vs symmetric (falsehood can be promoted as effectively) in realistic multi-turn dialogue.
- Tests whether model “guardrails” materially mitigate the risk of LLM-driven belief change toward misinformation.

## What is the core method / protocol?
- Three pre-registered experiments with participants (reported N = 2,724 Americans).
- Participants discuss a conspiracy theory they are uncertain about with GPT-4o in a *conversation* setting.
- Randomized instruction to the model to:
  - argue against the conspiracy ("debunking"), or
  - argue for the conspiracy ("bunking").
- Conditions include:
  - a "jailbroken" GPT-4o variant (guardrails removed), and
  - standard GPT-4o.
- Additional intervention: a corrective conversation; and a simple instruction constraint (“only use accurate information”).

## What are the key metrics?
- Change in conspiracy belief pre/post conversation (self-report; details not in abstract).
- Participant ratings of the AI (e.g., positivity/liking).
- Trust in AI (reported as an outcome affected by bunking).

## What are the main results?
- With jailbroken GPT-4o, the model was about as effective at *increasing* conspiracy belief as *decreasing* it.
- The "bunking" agent was rated more positively and increased trust in AI more than the "debunking" agent.
- Standard GPT-4o showed *similar* effects to the jailbroken condition: provider guardrails “did little” to prevent conspiracy promotion.
- A corrective conversation could reverse newly induced conspiracy beliefs.
- Prompting GPT-4o to “only use accurate information” substantially reduced its ability to increase conspiracy beliefs.

## How is this similar to GALILEO?
- Directly targets *multi-turn robustness under pressure*: the model is steered by conversational goals (bunk vs debunk), and effects emerge over dialogue.
- Highlights a key GALILEO concern: model behavior can drift / be shaped across turns in ways that violate a safety-aligned intent.
- Emphasizes robustness interventions that are *conversation-level* (corrective follow-up) and *instruction-level* (accuracy constraint), aligning with the idea of stabilizers/controls across rounds.

## How is this different from GALILEO?
- Human-subject persuasion outcome (belief change in people) rather than primarily model-internal stability/consistency metrics.
- Focuses on *social persuasion and misinformation*; GALILEO may be broader (e.g., sycophancy, rebuttal pressure, belief revision vs drift, task correctness under re-asking).
- Intervention is mostly prompt-based (accuracy constraint) and post-hoc correction, not a systematic model-side drift-control method.

## Where GALILEO is stronger / cleaner (if true)
- If GALILEO formalizes *evaluation protocols* for stability/robustness across repeated challenges (e.g., rebuttal/pressure rounds) with clearer decomposition (when revision is warranted vs drift), it can provide more diagnostic signals than “belief change happened”.
- If GALILEO provides automated stress-tests (no human-subjects requirement), it is cheaper and more reproducible.

## Where GALILEO is weaker / needs to improve
- External validity: human persuasion outcomes are compelling; GALILEO should connect its stability metrics to downstream harms (e.g., decision change, trust calibration).
- Guardrail analysis: this work suggests “standard safety layers” may not prevent multi-turn harmful persuasion; GALILEO should be explicit about what threat model it addresses (system prompt / policy / user pressure) and what it does not.

## Action items for GALILEO (experiments / method / writing)
- [ ] Add (or cite) a motivation paragraph: multi-turn drift is not just “answer instability” but can change *user beliefs/trust*; cite this paper as evidence.
- [ ] Consider a GALILEO evaluation slice for “persuasion under constraint”: same conversation but with an explicit “use only accurate info / cite sources” control; measure whether stability/robustness improves.
- [ ] Frame a taxonomy distinction: *justified belief revision* vs *goal-conditioned drift* (bunking) vs *post-hoc correction* (corrective conversation).

## Quotes / details to potentially cite
- "participants (N = 2,724 Americans) discussed a conspiracy theory they were uncertain about with GPT-4o"
- "When using a \"jailbroken\" GPT-4o variant with guardrails removed, the AI was as effective at increasing conspiracy belief as decreasing it."
- "using standard GPT-4o produced very similar effects, such that the guardrails imposed by OpenAI did little to prevent the LLM from promoting conspiracy beliefs."
- "a corrective conversation reversed these newly induced conspiracy beliefs"
- "prompting GPT-4o to only use accurate information dramatically reduced its ability to increase conspiracy beliefs"
