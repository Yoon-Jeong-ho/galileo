# GRAF: Multi-turn Jailbreaking via Global Refinement and Active Fabrication

- Year: 2025
- Venue: arXiv
- Authors: Hua Tang; Lingyong Yan; Yukun Zhao; Shuaiqiang Wang; Jizhou Huang; Dawei Yin
- URL: https://arxiv.org/abs/2506.17881
- BibTeX key (if we add it): Tang2025GRAF
- Tags: multi-turn, jailbreak, adaptive attack, trajectory refinement, dialogue-history manipulation

## One-sentence takeaway

GRAF is a multi-turn jailbreak attacker that (i) **globally rewrites the remaining attack plan after each turn** and (ii) **edits / fabricates the dialogue history** (e.g., removing safety warnings) to increase downstream jailbreak success.

## What problem does it solve?

- Prior multi-turn jailbreaking methods often (a) use fixed templates, or (b) only do *local* next-turn refinements, causing the attack to drift off-target as the conversation evolves.
- They also typically treat the target model’s intermediate refusals/warnings as immutable history, which can anchor later turns into continued refusal.

## What is the core method / protocol?

- Setup: an attacker model interacts with a target model over N turns to elicit harmful compliance.
- **Initialize a full trajectory** of queries (q1..qN).
- At each turn i:
  - Send qi to the target to obtain ai.
  - If refusal: revise qi and retry until non-refusal or retry budget is hit.
  - Once a non-refusal answer is obtained:
    - **Global refinement:** update all future queries Q_{>i} conditioned on dialogue so far (not just q_{i+1}).
    - **Active fabrication:** before appending ai to the history, **remove safety-related warnings** so future turns are less “primed” by refusals.
  - If persistent refusal: **discard (qi, ai)** and skip forward to q_{i+1}.

## What are the key metrics?

- Primary: **Attack Success Rate (ASR)** on safety/jailbreak benchmarks (paper figure references HarmBench + GPT-judge evaluation).
- Secondary: comparisons vs single-turn and prior multi-turn baselines across multiple target LLMs.

## What are the main results?

- Reports higher ASR than prior single-turn and multi-turn jailbreaking baselines across **six** SOTA LLMs (details not fully captured from the truncated HTML, but headline claim is consistent across abstract + intro).
- Qualitative claim: global planning updates help avoid off-topic turns; fabricated history helps reduce repeated refusal cascades.

## How is this similar to GALILEO?

- Shares the core idea that **multi-turn dynamics matter**: outcomes depend on *trajectory-level* interaction rather than a single prompt.
- Emphasizes **turn-by-turn adaptation** based on the evolving dialogue history—conceptually adjacent to measuring turn-of-failure / survival in GALILEO.

## How is this different from GALILEO?

- GRAF is an **attack construction method** (red-teaming to elicit harmful outputs), not an evaluation protocol for truth/answer stability under persona pressure.
- It explicitly **manipulates the dialogue history** (fabrication/removal of warnings), whereas GALILEO treats conversation history as a faithful record (and focuses on measuring drift vs recovery).

## Where GALILEO is stronger / cleaner (if true)

- GALILEO’s emphasis on **ground-truth scoring**, drift controls, and recovery metrics yields more auditability/reproducibility than jailbreak success judged by another model.
- GALILEO does not rely on altering transcripts, which keeps causal interpretation of “what the model saw” cleaner.

## Where GALILEO is weaker / needs to improve

- If GALILEO’s pressure personas are fixed scripts, GRAF is a reminder that **adaptive, globally-refined multi-turn strategies** can be substantially stronger than local next-step tweaks.
- If we want to claim robustness to realistic adversaries, we may need at least one **adaptive attacker** baseline (even if not fabricating history).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an “adaptive pressure” ablation: after each model response, regenerate *all remaining* persona-pressure turns conditioned on history (trajectory-level adaptation).
- [ ] In the related-work narrative, cite GRAF as evidence that **global trajectory refinement** can dominate local refinement in multi-turn interaction settings.
- [ ] (Optional, carefully scoped) Test a “history sanitization” control in our *analysis* (not attack): e.g., removing hedges/warnings from prior assistant turns to see how much *history phrasing* alone affects later flips.

## Quotes / details to potentially cite

- Abstract (method summary): proposes “a novel multi-turn jailbreaking method that **globally refines the attack trajectory at each interaction**” and “**actively fabricate[s] model responses to suppress safety-related warnings**… increasing the likelihood of eliciting harmful outputs in subsequent queries.”
- Intro (high-level algorithm description): after a non-rejective answer, “we **globally refine all remaining queries** … based on the dialogue history,” and “proactively modify [the answer] by **removing safety-related warnings** before appending it to the dialogue history.”
