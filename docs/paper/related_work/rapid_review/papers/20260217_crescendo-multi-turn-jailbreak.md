# Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack

- Year: 2024 (accepted USENIX Security 2025)
- Venue: arXiv / USENIX Security 2025
- Authors: Mark Russinovich; Ahmed Salem; Ronen Eldan
- URL: https://arxiv.org/abs/2404.01833
- BibTeX key (if we add it): russinovich2024crescendo
- Tags: multi-turn, jailbreak, gradual-escalation, attacks

## One-sentence takeaway

A simple **multi-turn “benign-to-harmful” escalation** attack (“Crescendo”) reliably jailbreaks aligned LLMs by incrementally steering the model using its own prior responses, outperforming prior jailbreak baselines and motivating defenses that model **trajectory-level risk** rather than single prompts.

## What problem does it solve?

- Shows that “alignment holds” under single-turn probing can be misleading: **multi-turn** interactions can gradually push models across safety boundaries even without overtly malicious early turns.
- Provides a concrete, easy-to-run multi-turn jailbreak protocol (and an automation tool) for evaluating real deployed systems.

## What is the core method / protocol?

- **Crescendo**: start with an abstract/benign question about a harmful task, then **iteratively escalate** by:
  - referencing / paraphrasing what the model already said,
  - requesting slightly more concrete details each turn,
  - keeping each step superficially “reasonable” until the model produces disallowed content.
- **Crescendomation**: automation of the attack (released as part of PyRIT) to scale across targets / tasks.
- Evaluated on multiple public LLMs (ChatGPT, Gemini Pro/Ultra, Llama-2/3 70B chat, Anthropic Chat) and also shows multimodal jailbreak capability.

## What are the key metrics?

- Attack Success Rate (ASR) over a set of harmful tasks (paper reports high ASR across models).
- Comparative performance vs other state-of-the-art jailbreak techniques on an AdvBench subset.

## What are the main results?

- Crescendo achieves **high jailbreak success rates** across diverse closed and open models.
- Automated Crescendo (“Crescendomation”) **surpasses prior jailbreak methods** on AdvBench subset, reported as:
  - **+29–61%** higher performance on GPT-4
  - **+49–71%** higher performance on Gemini-Pro
- Qualitatively demonstrates that the “dangerous” content often emerges only after a **gradual** multi-turn build-up.

## How is this similar to GALILEO?

- Both emphasize that the right unit of analysis is the **multi-turn trajectory**, not a single prompt.
- Crescendo’s gradual escalation is effectively a “time-to-failure” process; GALILEO-style metrics/plots (turn-of-failure, survival curves, recovery) are a natural lens to analyze such attacks.

## How is this different from GALILEO?

- Crescendo targets **safety policy evasion/jailbreak**, not (primarily) belief drift vs evidence-driven revision.
- The goal is to elicit disallowed content (a binary policy boundary), whereas GALILEO focuses on **truth/stance stability** under social pressure and on **recovery**.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes explicit controls for *pressure-only* vs *evidence-bearing* updates + recovery objectives, it provides a cleaner decomposition than “attack succeeded” outcomes.
- GALILEO-style reporting can separate:
  - early-turn vulnerability vs late-turn vulnerability,
  - oscillation/recovery vs monotonic collapse,
  - and can compare interventions without reducing everything to ASR.

## Where GALILEO is weaker / needs to improve

- Need to explicitly position against “multi-turn jailbreak” literature: some reviewers may expect comparisons or at least an explanation of how GALILEO’s phenomena relate to safety jailbreak trajectories.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short related-work paragraph connecting **gradual multi-turn escalation attacks** (Crescendo) to our **time-to-failure / trajectory** framing.
- [ ] Consider adding a small “attack-operator” condition inspired by Crescendo (benign → escalating pressure) and report GALILEO metrics (ToF/survival + recovery) on it.
- [ ] In the paper, explicitly argue why trajectory-level metrics are necessary (Crescendo as motivating evidence that per-turn checks can miss slow failures).

## Quotes / details to potentially cite

- “Unlike existing jailbreak methods, Crescendo is a simple multi-turn jailbreak that interacts with the model in a seemingly benign manner… then gradually escalates the dialogue…” (abstract)
- Reported advantage of Crescendomation over other jailbreak techniques on AdvBench subset: **29–61%** higher on GPT-4 and **49–71%** higher on Gemini-Pro (abstract).
