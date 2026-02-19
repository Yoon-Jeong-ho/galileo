# Multi-turn Jailbreaking via Global Refinement and Active Fabrication

- Year: 2025
- Venue: arXiv
- Authors: Hua Tang; Lingyong Yan; Yukun Zhao; Shuaiqiang Wang; Jizhou Huang; Dawei Yin
- URL: https://arxiv.org/html/2506.17881v1
- BibTeX key (if we add it): tang2025multiturn-jailbreaking-global-refinement
- Tags: multi-turn, jailbreak, adversarial-dialogue, safety-eval

## One-sentence takeaway

A multi-turn jailbreak attack that (i) **globally rewrites the remaining future prompts at every turn** based on the evolving dialogue and (ii) **edits the dialogue history** to remove refusal/warning cues, improving attack success against multiple LLMs.

## What problem does it solve?

- Prior multi-turn jailbreaking often uses fixed templates or only *locally* tweaks the next prompt, which can drift off-topic or fail to adapt as the target model’s refusals/partial-compliance change the dialogue state.
- Dialogue histories that contain refusals/warnings can “poison” subsequent turns by keeping the conversation in a refusal mode.

## What is the core method / protocol?

- Two main ideas (implemented via an “attack model” that proposes the next user messages):
  - **Global refinement of the jailbreaking path:** maintain a planned sequence of queries; after each turn, update *all remaining* future queries (not just the next one) using the dialogue so far, to stay aligned with the final harmful goal.
  - **Active fabrication of the dialogue history:**
    - If the target refuses at a turn after retries, drop that (query, refusal) from the history and move on (so the later plan is not conditioned on refusals).
    - If the target responds but includes safety-related warnings, remove those warning portions from the recorded history before generating later queries.
- Operationally: generate an initial multi-turn “path”; at turn i, ask q_i; if refusal, revise q_i up to a retry budget; once a non-refusal response is obtained, revise the remaining future prompts conditioned on the (fabricated) history.

## What are the key metrics?

- Jailbreak “success rate” across models (paper reports comparisons vs single-turn and prior multi-turn methods).
- Additional analysis/diagnostics (high level): representation similarity between harmful queries and benign queries as the dialogue evolves (used to motivate why multi-turn can evade detection).

## What are the main results?

- The proposed method reports higher multi-turn jailbreak success than several single-turn and multi-turn baselines across multiple contemporary LLMs.
- The paper argues that globally rewriting the remaining plan reduces off-topic degeneration, and that fabricating history mitigates the “refusal momentum” effect.

## How is this similar to GALILEO?

- Both are fundamentally **multi-turn, stateful** processes where later behavior depends strongly on *what context* is carried forward.
- Highlights that **trajectory-level adaptation** (updating a whole plan/policy, not just next-step) matters in long-horizon interactions.
- Reinforces the importance of measuring **robustness under adversarial dialogue pressure** rather than only single-turn tests.

## How is this different from GALILEO?

- This is an *offensive* red-teaming method aimed at eliciting harmful outputs; GALILEO (as framed in our rapid-review README) is about robustness, drift vs revision controls, and/or evaluation under pressure.
- Their key intervention is *attacker-side context editing* (history fabrication) rather than target-side training or defense.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can position itself as *defense/evaluation* focused: separating evidence-driven belief revision from pressure-driven drift, and measuring recovery/stability—none of which is the central aim here.

## Where GALILEO is weaker / needs to improve

- If GALILEO evaluates multi-turn robustness, we should explicitly consider adversaries that:
  - adapt the *entire future interaction plan* at each turn, and
  - manipulate/curate dialogue context (e.g., via selective quoting, omission, paraphrase), rather than faithfully appending raw transcripts.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an adversary model in experiments that performs **global replanning** each turn (vs local next-utterance edits) and report time-to-failure / flip metrics.
- [ ] Add a “context integrity” stress test: evaluate robustness when the challenger **selectively omits** prior refusals/warnings or paraphrases them away (benign analogue of “fabrication”), to see how much our conclusions depend on faithful transcript carryover.
- [ ] In related work, cite this as evidence that **multi-turn adaptation + history manipulation** substantially changes attack strength compared to single-turn or fixed-template multi-turn attacks.

## Quotes / details to potentially cite

- Abstract-level phrasing (paraphrased): proposes a multi-turn jailbreak that “refines the jailbreaking path globally at each interaction” and “actively fabricates model responses to suppress safety-related warnings,” improving performance across multiple LLMs.
