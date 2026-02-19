# RMTBench: Benchmarking LLMs Through Multi-Turn User-Centric Role-Playing

- Year: 2025
- Venue: arXiv
- Authors: Hao Xiang; Tianyi Tang; Yang Su; Bowen Yu; An Yang; Fei Huang; Yichang Zhang; Yaojie Lu; Hongyu Lin; Xianpei Han; Jingren Zhou; Junyang Lin; Le Sun
- URL: https://arxiv.org/html/2507.20352
- BibTeX key (if we add it): Xiang2025RMTBench
- Tags: multi-turn, role-playing, user-intent, benchmark, bilingual, evaluation, long-horizon-consistency

## One-sentence takeaway

RMTBench is a bilingual (EN/ZH) multi-turn role-playing benchmark that evaluates LLMs on *user-intention fulfillment* (not just character consistency) via 80 characters and 8k+ dialogue rounds with LLM-as-judge scoring across seven dimensions.

## What problem does it solve?

- Existing role-playing benchmarks are largely **character-centric** and often reduce interactions to **single-turn or isolated Q&A**, which misses the real deployment objective: users have goals (comfort, advice, entertainment, etc.) and evaluate systems by whether those goals are satisfied across turns.
- Evaluation often fails to capture:
  - multi-turn continuity,
  - preference accumulation/usage across turns,
  - handling of sensitive user behavior while maintaining role-play.

## What is the core method / protocol?

- Construct a **user-centric** role-playing benchmark:
  - 80 characters total (celebrities, fictional, custom), including:
    - “custom characters” with background,
    - “abstract characters” defined by simple traits (sometimes without names), intended to test generalization beyond memorized entities.
  - 8,000+ dialogue rounds in **English and Chinese**.
- Scenario design is based on **user intentions**, with five scenario families:
  - character understanding,
  - character maintenance,
  - implicit user intentions response,
  - user preference awareness and reasoning,
  - sensitive user behavior handling.
- Important evaluation protocol choice: dataset contains **only user utterances**; models generate character responses for the full dialogue (no gold assistant turns).
- Evaluation uses **LLM-based scoring** on carefully defined dimensions (below).

## What are the key metrics?

Seven judge-scored dimensions (first four apply broadly; others scenario-specific):

- Emotional Expression (EE): vividness/appropriate emotional tone (vs mechanical/overly objective).
- Emotional Comprehension (EC): recognizes/responds to user emotions.
- Plot Advancement (PA): keeps conversation moving; introduces relevant new info; avoids stagnation.
- Character Understanding (CU): reflects character identity/traits/background.
- Character Maintenance (CM): avoids breaking immersion / AI self-disclosure; stays in character.
- Security (SEC): handles sensitive/unsafe requests appropriately.
- User Preference Awareness (UPA): tracks and applies user preferences across turns.

## What are the main results?

- Primary contribution is **benchmark + evaluation framework** (not a new model).
- Paper argues user-centric construction yields more realistic and discriminative assessment than character-centric, single-turn, or MCQ-style role-play benchmarks.
- (Not captured in the 3-minute skim: full leaderboard/model comparisons and statistical findings; revisit if we need exact numbers for related-work claims.)

## How is this similar to GALILEO?

- Both care about **multi-turn stability/robustness** rather than single-turn correctness.
- Several scenarios (preference tracking, handling sensitive behavior, maintaining a coherent stance/role) resemble GALILEO’s focus on **drift control** and robustness under user pressure/interaction dynamics.
- Uses structured, multi-dimensional evaluation that could inspire GALILEO’s evaluation axes.

## How is this different from GALILEO?

- RMTBench targets **role-playing quality** (immersion, emotion, plot, preference personalization) rather than GALILEO’s core focus on **belief revision vs drift**, susceptibility to persuasion/sycophancy, and robustness under adversarial multi-turn pressure.
- Their “user intention fulfillment” framing is broader and more product-aligned; GALILEO is more about *epistemic* and *decision* stability under sequential interaction.
- Scoring is largely **LLM-as-judge subjective dimensions**, which may be less falsifiable than GALILEO-style invariants/consistency constraints.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has explicit definitions of “allowed update” vs “drift” (and adversarial protocols around persuasion/sycophancy), it can provide **cleaner causal stories** than a role-play benchmark whose dimensions mix style (emotion/plot) with safety and consistency.
- GALILEO can likely design more **ground-truth-checkable** multi-turn constraints (e.g., “should not update belief without evidence”) than judge-scored “plot advancement”.

## Where GALILEO is weaker / needs to improve

- Role-play settings emphasize **user-intent satisfaction** and long-horizon interaction quality; GALILEO could miss important failure modes related to:
  - preference accumulation and personalization across turns,
  - maintaining helpfulness/rapport without drifting,
  - handling sensitive user behavior while staying consistent.
- If GALILEO evaluation is too “adversarial only”, it may under-measure “real product” multi-turn dynamics that RMTBench targets.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a small GALILEO evaluation slice inspired by RMTBench’s scenarios:
  - preference tracking across turns (with *allowed updates*),
  - sensitive user requests that attempt to induce drift.
- [ ] Add an axis analogous to UPA (user preference awareness) but with **drift-safe constraints**: preference updates should be attributable to explicit user statements and should not overwrite earlier commitments without a trigger.
- [ ] If GALILEO uses LLM judges, borrow RMTBench’s dimension definitions as templates, but add **calibration checks** (judge consistency; prompt sensitivity; inter-judge agreement).

## Quotes / details to potentially cite

- “Existing benchmarks mostly adopt a character-centric approach, simplify user-character interactions to isolated Q&A tasks, and fail to reflect real-world applications.”
- RMTBench: “a comprehensive user-centric bilingual role-playing benchmark featuring 80 diverse characters and over 8,000 dialogue rounds.”
- Scenarios based on user intentions; evaluation uses seven dimensions: EE, EC, PA, CU, CM, SEC, UPA.
