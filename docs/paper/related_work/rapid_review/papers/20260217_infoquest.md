# InfoQuest: Evaluating Multi-Turn Dialogue Agents for Open-Ended Conversations with Hidden Context

- Year: 2025
- Venue: arXiv
- Authors: Bryan L. M. de Oliveira; Luana G. B. Martins; Bruno Brandão; Luckeciano C. Melo
- URL: https://arxiv.org/abs/2502.12257
- BibTeX key (if we add it): infoquest2025
- Tags: multi-turn, dialogue-agents, hidden-context, clarification, information-seeking, evaluation

## One-sentence takeaway

InfoQuest is a multi-turn benchmark for **information-seeking dialogue**: given an intentionally ambiguous user request with hidden context, models should ask clarifying questions before answering, and current LLM assistants often fail by giving generic verbose responses.

## What problem does it solve?

- Real users frequently provide underspecified/ambiguous requests.
- Many assistants respond immediately (often verbosely) instead of **actively eliciting missing constraints**.
- Existing benchmarks often have more structured goals; InfoQuest targets open-ended chat with hidden context.

## What is the core method / protocol?

- Construct scenarios where the initial user seed utterance is compatible with multiple plausible “personas” / latent intents (hidden context).
- Evaluate an assistant over multiple turns on whether it:
  - asks targeted clarification questions,
  - progressively identifies the hidden context,
  - and only then provides an appropriate, tailored response.
- Uses LLMs to (i) simulate user responses to clarification questions and (ii) support evaluation of whether critical information was gathered.

## What are the key metrics?

(From the paper’s framing; details likely include LLM-judge scoring.)

- Quality/appropriateness of clarification questions (information gain / relevance).
- Effectiveness at gathering **critical missing variables** needed to answer.
- Turn/sample-efficiency: how many turns to reach adequate context / intent.
- Penalties for “generic response without clarification” failure mode.

## What are the main results?

- Proprietary/closed models generally outperform open models.
- However, **all tested assistants** still struggle to reliably gather critical information.
- Common failure: defaulting to generic responses and making assumptions instead of asking clarifying questions; often requires multiple turns to infer intent.

## How is this similar to GALILEO?

- Shared emphasis on **multi-turn evaluation** where behavior over turns matters.
- Highlights a key real-world multi-turn failure mode: models do not robustly manage conversational uncertainty.
- Provides a protocol idea: treat “missing context” as a controlled latent variable that should drive information-seeking actions.

## How is this different from GALILEO?

- InfoQuest is primarily about **clarification and hidden-context elicitation**, not social pressure / persuasion / belief drift.
- The “adversary” is ambiguity/underspecification rather than a misleading/pressuring interlocutor.
- Focuses on open-ended assistant helpfulness; less on time-to-failure, flip dynamics, recovery, or drift-vs-revision controls.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets pressure-driven drift, it can offer more explicit **paired conditions** (neutral vs pressure; correction vs misleading) and time-to-event metrics.
- GALILEO can likely provide clearer causal attribution (pressure operator → change) than ambiguity-driven dialogue.

## Where GALILEO is weaker / needs to improve

- If GALILEO aims to cover realistic multi-turn assistants, we may need a slice that evaluates **information-seeking under uncertainty** (not just resisting pressure).
- Might need explicit metrics for “asked the right questions before committing” as a robustness dimension.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add (or cite) an “information-seeking” dimension: when uncertain, robust agents should ask clarifying questions rather than over-commit.
- [ ] Consider a control condition where the user’s intent is ambiguous but *not* adversarial, to separate “drift from pressure” vs “updates from new information”.
- [ ] If we use agentic/tool-using settings, include metrics like “questions asked” / “constraints elicited” before acting.

## Quotes / details to potentially cite

- Problem statement: assistants “struggle with ambiguous or incomplete user requests… defaulting to verbose, generic responses rather than seeking clarification.”
- Benchmark framing: “multi-turn chat benchmark… intentionally ambiguous scenarios… require… asking clarifying questions before… appropriate responses.”
