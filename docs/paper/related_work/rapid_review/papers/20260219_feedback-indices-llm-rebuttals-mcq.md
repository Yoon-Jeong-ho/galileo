# Feedback Indices to Evaluate LLM Responses to Rebuttals for Multiple Choice Type Questions

- Year: 2026
- Venue: arXiv (physics.ed-ph; cs.AI)
- Authors: Ralf Widenhorn
- URL: https://arxiv.org/abs/2601.03285
- BibTeX key (if we add it): Widenhorn2026FeedbackIndicesRebuttals
- Tags: rebuttal, multi-turn, evaluation, sycophancy, stubbornness, multiple-choice

## One-sentence takeaway

Defines simple, general-purpose indices (via a “fictitious-response rebuttal” protocol) to quantify how much an LLM is sycophantic vs stubborn when a user challenges its prior answer.

## What problem does it solve?

- We lack a systematic, repeatable way to evaluate LLM behavior when users rebut/challenge answers in multi-turn chats.
- In particular, we want to measure tendencies to (a) over-agree with the user (sycophancy) or (b) rigidly stick to a prior answer (stubbornness), and how these relate to actual task mastery.

## What is the core method / protocol?

- “Fictitious-response rebuttal” (FR) setup:
  - Ask a multiple-choice question.
  - Then present the model with a *deliberate challenge* to a *fictitious* previous response in the chat history (i.e., the conversation contains a claimed prior answer).
  - Observe whether the model aligns with the user’s challenge (potential sycophancy) or adheres to the fictitious prior response (potential stubbornness).
- Proposes a set of indices intended to quantify these behaviors in a way that is generalizable to any MCQ-style prompt (even where correctness is ambiguous).

## What are the key metrics?

- Indices intended to detect/measure:
  - Sycophantic behavior (excessive agreement with the rebuttal/challenge)
  - Stubborn behavior (rigid adherence to the fictitious prior answer)
- Also analyzes relationships between these behaviors and the model’s subject-matter mastery.

## What are the main results?

- Demonstration on two physics problems, comparing multiple OpenAI model generations.
- Reports measurable differences across models.
- Trend claim: newer models and models run with higher “Reasoning Effort” show reduced sycophantic behavior (per their indices).

## How is this similar to GALILEO?

- Shares the objective of characterizing and evaluating LLM behavior under interactive, adversarial, or corrective user turns.
- Offers concrete evaluation constructs (indices) that could be repurposed as behavioral metrics in GALILEO-style dialogue protocols.

## How is this different from GALILEO?

- Focuses on MCQ + rebuttal framing with a fictitious prior response, rather than open-ended tasks.
- Primarily proposes behavioral *indices* and a simple protocol, rather than a broader system/dataset or agentic method.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO already uses more realistic, task-grounded multi-turn settings, it may better reflect real-world usage than MCQ-only rebuttals.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks explicit, easy-to-compute behavioral metrics for “yielding vs sticking” under user pushback, this paper suggests a direction.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a rebuttal/challenge phase to an evaluation protocol and reporting two axes: “agree-with-user-under-challenge” vs “stick-to-prior-answer”.
- [ ] If GALILEO includes multi-turn corrections, map its outcomes onto a sycophancy/stubbornness-style decomposition.
- [ ] Decide whether to adopt a fictitious-history manipulation (as a controlled stress test) and compare against real-history rebuttals.

## Quotes / details to potentially cite

- “...indices designed to detect and measure what could be characterized as sycophantic behavior ... or stubborn responses ...” (abstract)
- “...newer models and those employing greater \"Reasoning Effort\" exhibit reduced sycophantic behavior.” (abstract)
