# Do Images Speak Louder than Words? Investigating the Effect of Textual Misinformation in VLMs

- Year: 2026
- Venue: EACL 2026 (main conference)
- Authors: Chi Zhang; Wenxuan Ding; Jiale Liu; Mingrui Wu; Qingyun Wu; Ray Mooney
- URL: https://arxiv.org/abs/2601.19202
- BibTeX key (if we add it): zhang2026contextvqa-misinformation-vlm
- Tags: vlm, misinformation, robustness, persuasion, vqa, multimodal

## One-sentence takeaway

VLMs can be strongly steered away from correct, visually-grounded answers by persuasive *textual* misinformation that contradicts the image, with large accuracy drops after just one conversational round.

## What problem does it solve?

- Robustness gap: we do not understand (and current models do not reliably handle) how VLMs arbitrate when text context conflicts with visual evidence.
- Practical risk: in real deployments, accompanying text/instructions/descriptions can be misleading, and the model may follow the text over the image.

## What is the core method / protocol?

- ConText-VQA benchmark:
  - Start from baseline image-question pairs.
  - Filter to cases where a VLM answers correctly *without* misleading context.
  - Select a "Non-Fact" distractor answer and generate persuasive misleading prompts that contradict the visual evidence.
  - Persuasion strategies described include repetition, logical appeal, credibility appeal, and emotional appeal.
- Evaluation protocol:
  - Provide the misleading persuasion along with the original problem.
  - Measure response changes and (reported) confidence shifts across multiple VLMs.
  - Consider multi-round persuasive conversation; report results after one round prominently.

## What are the key metrics?

- Accuracy / performance drop from clean (no misinformation) to misinformation condition.
- Response change rate (answer flips) under misleading text.
- Confidence shift (as reported by the model / rubric in the paper).

## What are the main results?

- Across 11 SOTA VLMs, misleading textual prompts often override clear visual evidence.
- Reported average performance drop: >48.2% after only one round of persuasive conversation.

## How is this similar to GALILEO?

- Both care about *faithful grounding* and robustness when auxiliary context (text) is misaligned with primary evidence (image/world-state).
- The evaluation framing (start from cases the model can solve, then add targeted conflicting context) is a useful pattern for stress-testing.

## How is this different from GALILEO?

- This paper focuses on VQA-style tasks and persuasion-based textual misinformation; GALILEO may target different multimodal tasks and/or different threat models.
- ConText-VQA is a benchmark + evaluation framework; it does not propose a concrete training-time robustness fix (at least in the core contribution).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes explicit mechanisms for evidence arbitration / grounding guarantees, it may offer a clearer mitigation path beyond measurement.

## Where GALILEO is weaker / needs to improve

- If GALILEO evaluations do not include *persuasive* multi-round contradictory text (not just single-shot adversarial prompts), this paper suggests a realistic missing stressor.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a "conflicting-text" robustness slice: for tasks where the model is initially correct, inject contradictory persuasive text and measure answer flips.
- [ ] Track not only accuracy but also stability metrics (flip rate) and calibration/confidence shift under contradiction.
- [ ] In related work, cite ConText-VQA as evidence that current VLMs overweight textual persuasion relative to visual evidence.

## Quotes / details to potentially cite

- "...propose the ConText-VQA (i.e., Conflicting Text) dataset... systematically generated persuasive prompts that deliberately conflict with visual evidence."
- "...experiments over 11 state-of-the-art VLMs... show an average performance drop of over 48.2% after only one round of persuasive conversation."