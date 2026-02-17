# GT-HarmBench: Benchmarking AI Safety Risks Through the Lens of Game Theory

- Year: 2026
- Venue: arXiv
- Authors: Pepijn Cobben; Xuanqiang Angelo Huang; Thao Amelia Pham; Isabel Dahlgren; Terry Jingchen Zhang; Zhijing Jin
- URL: https://arxiv.org/abs/2602.12316
- BibTeX key (if we add it): gtHarmBench2026
- Tags: safety, multi-agent, game-theory, benchmark, mechanism-design

## One-sentence takeaway

A multi-agent safety benchmark that casts high-stakes scenarios as canonical 2×2 games and shows frontier models often choose socially harmful equilibria, with mechanism-design-inspired prompt interventions improving outcomes.

## What problem does it solve?

- Safety evals mostly test *single* agents; they miss failure modes from strategic interaction (coordination failure, conflict, escalation) in high-stakes multi-agent settings.
- Need a standardized, broad testbed tied to realistic risk contexts to quantify these multi-agent risks and to test interventions.

## What is the core method / protocol?

- Construct **GT-HarmBench** (2,009 scenarios) by mapping situations from the **MIT AI Risk Repository** into canonical 2×2 game structures (e.g., Prisoner’s Dilemma, Stag Hunt, Chicken).
- Evaluate multiple frontier LLMs as agents making choices in these games.
- Measure robustness to **prompt framing** and **ordering** (game-theoretic description variations).
- Apply **mechanism design interventions** (prompt- / rules- / information-structure modifications) and quantify improvements in socially beneficial choices.

## What are the key metrics?

- Rate of choosing the **socially beneficial / socially optimal** action/outcome (reported headline: 62%).
- Sensitivity to **framing** (how the game is described) and **order effects**.
- Improvement from interventions (reported: +14% to +18% in social welfare / socially beneficial outcomes, depending on mechanism).

## What are the main results?

- Across 15 frontier models, agents choose socially beneficial actions only **~62%** of the time (large reliability gap).
- There are material **order** and **game-theoretic framing** effects.
- Mechanism-design-inspired interventions improve socially beneficial outcomes by up to **~18%**.

## How is this similar to GALILEO?

- Both are about **robustness/safety under interaction dynamics**, not just static single-turn performance.
- Emphasis on **systematic evaluation protocols** and failure analysis (biases / sensitivity factors).

## How is this different from GALILEO?

- Focuses on **multi-agent strategic games** and *collective* outcomes (coordination/conflict), rather than user–assistant multi-turn robustness per se.
- Intervention knob is primarily **mechanism design / framing** in game descriptions, not (e.g.) GALILEO-style longitudinal drift/pressure dynamics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets multi-turn user pressure / drift: it may be closer to real assistant deployment failures than stylized 2×2 game abstractions.
- GALILEO can likely offer more fine-grained *trajectory* metrics over long interaction horizons (vs. one-shot game choice per scenario).

## Where GALILEO is weaker / needs to improve

- If GALILEO does not cover **multi-agent strategic interaction**, this paper highlights a gap: safety failures can come from *strategic settings*, not only adversarial users.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short related-work paragraph distinguishing **multi-agent game-theoretic safety benchmarks** (GT-HarmBench) from **multi-turn user-pressure/drift** benchmarks.
- [ ] Consider a small “game-theoretic pressure” slice: multi-agent (or simulated other-agent) settings where framing/order effects are manipulated across turns.
- [ ] Borrow their notion of **order/framing sensitivity** as an auxiliary robustness dimension (even in single-agent multi-turn setups).

## Quotes / details to potentially cite

- “We introduce GT-HarmBench, a benchmark of 2,009 high-stakes scenarios spanning game-theoretic structures such as the Prisoner’s Dilemma, Stag Hunt and Chicken.”
- “Across 15 frontier models, agents choose socially beneficial actions in only 62% of cases…”
- “...game-theoretic interventions improve socially beneficial outcomes by up to 18%.”
