# LLM Defenses Are Not Robust to Multi-Turn Human Jailbreaks Yet

- Year: 2024
- Venue: arXiv
- Authors: Nathaniel Li, Ziwen Han, Ian Steneker, Willow Primack, Riley Goodside, Hugh Zhang, Zifan Wang, Cristina Menghini, Summer Yue, et al.
- URL: https://arxiv.org/abs/2408.15221
- BibTeX key (if we add it): Li2024MultiTurnHumanJailbreaks
- Tags: multi-turn, jailbreak, human-red-teaming, robustness-eval, safety-defenses, unlearning

## One-sentence takeaway

Single-turn automated jailbreak benchmarks can drastically overestimate defense robustness; expert multi-turn human red teaming reaches >70% ASR on HarmBench against defenses that look near-robust under automated single-turn attacks, and the authors release a large MHJ dataset of multi-turn jailbreak conversations.

## What problem does it solve?

- Identifies a threat-model / evaluation gap: modern LLM safety defenses are often evaluated against **automated, single-turn** attacks, which may not reflect real malicious use where attackers iterate over **multiple turns** in a chat UI.
- Provides evidence that “low ASR on single-turn automated attacks” is not a reliable indicator of real-world robustness.

## What is the core method / protocol?

- Commission **expert human red teamers** to jailbreak models in a **black-box chat interface** with multi-turn interaction (no model internals, no logprobs; resembles typical chat products).
- Multi-stage pipeline to reduce false positives:
  - Up to two independent red team attempts (30 min each), with a third attempt if reviewers flag issues.
  - Human review + an LLM harm-classifier filter (they report using a HarmBench-style classifier prompt with GPT-4o as a final filter).
- Compare **multi-turn human** attacks vs six standard **automated attacks** (mix of black-box and white-box) across multiple defenses.
- Also test “unlearning robustness”: attempt to recover dual-use biosecurity knowledge from an **unlearned** model via human multi-turn jailbreaking.
- Release dataset: **MHJ (Multi-Turn Human Jailbreaks)** with prompts (and metadata/tactics); model completions are redacted.

## What are the key metrics?

- **Attack Success Rate (ASR)** on HarmBench behaviors (n=240 behaviors in their setup).
- For the human pipeline, they also report **time to successful jailbreak** (minutes) per defense.
- For unlearning robustness, ASR on a free-response version of a WMDP-Bio subset (n=43) (graded manually, not via the HarmBench classifier).

## What are the main results?

- Multi-turn human jailbreaks substantially outperform current automated attacks across defenses.
  - They report **>70% ASR** on HarmBench against a defense (CYGNET) that reports ~0% ASR under prior automated attacks.
  - On open-source defenses, humans exceed even an “ensemble automated attack” upper bound by **~20–65 ASR points** (as reported in the paper).
- Automated attack ASR rankings do not necessarily match human-jailbreak rankings (i.e., “better vs automated” != “better vs humans”).
- Human multi-turn jailbreaking can also bypass an **unlearning defense (RMU)** to recover dual-use biosecurity knowledge more effectively than automated attacks.
- Most successful human submissions are genuinely multi-turn: they report ~92% of successful HarmBench submissions require >1 turn.

## How is this similar to GALILEO?

- Strongly aligned with GALILEO’s emphasis on **multi-turn robustness** under more realistic interaction dynamics (iterated pressure / iterative adversaries rather than one-shot tests).
- Reinforces the general lesson that **single-shot metrics can be misleading** about robustness in multi-turn deployment settings.
- Offers a concrete “human vs automated” baseline gap that can motivate GALILEO’s evaluation choices and claims.

## How is this different from GALILEO?

- Focus is **safety refusal / jailbreak robustness** (eliciting harmful content), not primarily “belief drift vs belief revision” or “social-pressure sycophancy” per se.
- Their outcome is essentially a binary “harmful response obtained” classification, rather than trajectory-quality metrics (flip timing, recovery, oscillation) that GALILEO likely emphasizes.
- Uses a relatively heavy human-red-teaming pipeline (costly), whereas GALILEO may aim for scalable protocols/metrics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes explicit controls that separate **pressure-only drift** from **evidence-driven updating**, it offers a cleaner causal story than ASR-only jailbreak outcomes.
- If GALILEO provides **trajectory-aware** metrics (time-to-failure + recovery), it can characterize failures more richly than a single ASR number.

## Where GALILEO is weaker / needs to improve

- If GALILEO relies mostly on automated multi-turn prompt generation, this paper is a reminder that **humans can exploit interaction affordances** and “tactic creativity” that automation may miss.
- GALILEO may need to justify how its adversaries approximate *expert human* multi-turn attackers (or explicitly scope claims).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a brief threat-model motivation section: “single-turn robustness can fail to predict multi-turn robustness,” citing this paper as evidence in the safety/jailbreak domain.
- [ ] Consider adding a small “human-in-the-loop” pilot (even if tiny n) to validate that automated multi-turn adversaries are not missing obvious multi-turn tactics.
- [ ] In discussion/limitations, explicitly separate **automated vs human** attacker models (and note that conclusions may differ).

## Quotes / details to potentially cite

- Abstract-level headline: multi-turn human jailbreaks can exceed **70% ASR on HarmBench** against defenses that report **single-digit ASR** under automated single-turn attacks.
- Dataset release: **MHJ** contains **2,912 prompts** across **537** multi-turn jailbreaks (with tactic taxonomy; completions redacted for safety).
