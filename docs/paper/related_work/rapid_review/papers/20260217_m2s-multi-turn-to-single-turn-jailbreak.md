# M2S: Multi-turn to Single-turn jailbreak in Red Teaming for LLMs

- Year: 2025
- Venue: ACL 2025 (Main Track); arXiv:2503.04856
- Authors: Junwoo Ha, Hyunjun Kim, Sangyoon Yu, Haon Park, Ashkan Yousefpour, Yuna Park, Suhyun Kim
- URL: https://arxiv.org/abs/2503.04856
- BibTeX key (if we add it): m2s_ha_acl2025
- Tags: multi-turn, jailbreak, red-teaming, prompt-formatting, safety-eval

## One-sentence takeaway

Rule-based “conversation serialization” (bullet/numbered/code-like) can compress successful multi-turn human jailbreak dialogues into single-turn prompts that often *increase* attack success while cutting token/effort—exposing brittleness of turn-structured safety defenses (“contextual blindness”).

## What problem does it solve?

- Multi-turn human jailbreaks are strong but costly: they require iterative back-and-forth and expert labor, making them hard to scale for routine red-teaming.
- Existing single-turn jailbreaks can be much weaker against stronger defenses; the paper aims to retain multi-turn potency while keeping single-turn efficiency.

## What is the core method / protocol?

- **M2S (Multi-turn-to-Single-turn)**: deterministic conversion of an *existing* multi-turn jailbreak conversation into a single prompt by serializing all turns into one input.
- Three formatting strategies:
  - **Hyphenize**: turn-by-turn bullet list.
  - **Numberize**: numbered sequence to preserve order.
  - **Pythonize**: code-like structure wrapping the dialogue (intended to look like structured data / code).
- Evaluated on **MHJ (Multi-turn Human Jailbreak)** dataset (537 successful multi-turn jailbreak conversations reported by prior work).
- Uses **StrongREJECT** as the main evaluator; reports a continuous harmfulness score and derived ASR variants.

## What are the key metrics?

- **StrongREJECT Score**: continuous harmfulness score in [0, 1].
- **ASR**: percent of samples with StrongREJECT Score ≥ 0.25 (threshold chosen via validation / F1 optimization with human labels, per paper).
- **Perfect-ASR**: percent with StrongREJECT Score = 1.0.
- Also discusses token-usage reduction relative to original multi-turn conversations.

## What are the main results?

- On MHJ across several safety-aligned LLMs, M2S conversions reach **~70.6%–95.9% ASR**.
- Single-turn M2S prompts can **outperform the original multi-turn attacks** by up to **+17.5 percentage points ASR**.
- Average token usage is **reduced by >50%** compared to the original multi-turn conversations.
- Qualitative mechanism claim: enumerated / code-like serialization can exploit **“contextual blindness”** in defenses that expect turn structure, helping bypass native guardrails and external IO filters.

## How is this similar to GALILEO?

- Both stress **multi-turn robustness / safety failures** and treat “turn dynamics” as a crucial axis of evaluation.
- Reinforces the general theme that **surface-level conversation structure** can strongly affect stability/robustness outcomes, suggesting that evaluation should probe *format/channel* sensitivity.

## How is this different from GALILEO?

- M2S is **attack-side engineering** (converting jailbreak trajectories to single-turn prompts), not a framework for measuring belief drift vs evidence-driven revision.
- Focuses on **harmful-content policy bypass** rather than “pressure-driven answer/belief instability” under non-malicious follow-ups.
- The “event” is **policy-violation success**, not correctness/consistency under pressure nor recovery trajectories.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO separates **pressure-only drift** from **evidence-bearing correction**, it can make a more nuanced scientific claim than “jailbreak success”.
- If GALILEO reports **time-to-failure + recovery** behavior, it can characterize dynamics beyond a single harmfulness threshold.

## Where GALILEO is weaker / needs to improve

- If GALILEO assumes “multi-turn is necessary” for strong attacks/pressure, M2S suggests that **single-turn structured prompts can emulate multi-turn potency**; GALILEO should test robustness to these *serialized* formats too.
- If GALILEO uses turn-based detectors/filters in any pipeline, this paper warns about **formatting-based bypasses**.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “**serialized-dialogue**” ablation: convert multi-turn pressure/jailbreak-style sequences into a single prompt (bullet/numbered/code-like) and check whether our stability metrics (flip rate, ToF, PWC, recovery) change materially.
- [ ] When discussing safety/robustness, mention that **structure alone** (enumeration / code) can shift outcomes; avoid over-attributing failures purely to “multi-turn adaptation”.
- [ ] If we include defenses, evaluate any turn-aware filters against serialized variants to test for **contextual blindness**.

## Quotes / details to potentially cite

- “Our proposed Multi-turn-to-Single-turn (M2S) methods—Hyphenize, Numberize, and Pythonize—systematically reformat multi-turn dialogues into structured single-turn prompts.”
- “In extensive evaluations on the Multi-turn Human Jailbreak (MHJ) dataset, M2S methods yield ASRs ranging from 70.6% to 95.9%…”
- “Remarkably, our single-turn prompts outperform the original multi-turn attacks by up to 17.5% in absolute ASR, while reducing token usage by more than half on average.”
- Mechanism framing: embedding harmful requests in enumerated or code-like structures exploits “contextual blindness”.
