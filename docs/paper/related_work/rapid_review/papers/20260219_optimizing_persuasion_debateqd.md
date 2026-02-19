# Optimizing for Persuasion Improves LLM Generalization: Evidence from Quality-Diversity Evolution of Debate Strategies

- Year: 2025
- Venue: arXiv
- Authors: Aksel Joonas Reedi; Corentin Leger; Julien Pourcel; Loris Gaven; Perrine Charriau; Guillaume Pourcel
- URL: https://arxiv.org/abs/2510.05909
- BibTeX key (if we add it): reedi2025optimizing
- Tags: persuasion, debate, quality-diversity, prompt-evolution, generalization, RLVR-overfitting

## One-sentence takeaway

Evolving *persuasion*-optimized debate prompting strategies via a minimal quality-diversity tournament yields smaller train–test generalization gaps than *truth*-optimized variants while matching or improving test accuracy.

## What problem does it solve?

- Truth-/label-optimized LLM training and optimization (e.g., RL with verifiable rewards) can overfit: brittle reasoning patterns, boundary collapse, and degraded performance on harder examples.
- Prior persuasion/debate work suggests benefits, but comparisons are often confounded (different protocols, populations, or setups), making it unclear whether the *objective* (persuasion vs truth) is what drives generalization.

## What is the core method / protocol?

- **Information-asymmetric debate** on QuALITY (HARD subset): debaters see the source text; the judge only sees the debate transcript.
- **DebateQD**: a minimal Quality-Diversity evolutionary loop over *prompts/strategies* (not separate model weights).
  - Initialize strategy prompts across multiple “families” (e.g., rationality, authority, emotional appeal, etc.).
  - Run Swiss-style tournaments; estimate Elo ratings.
  - Selection + mutation via an LLM “mutator” to generate improved prompt variants.
  - Repeat for multiple generations; evaluate top strategies.
- **Controlled objective swap**: keep the debate protocol fixed and change only the fitness definition:
  - **Persuasion objective**: individual strategies rewarded for winning (convincing the judge), irrespective of ground-truth truth.
  - **Truth objective**: strategies rewarded for *collaborative correctness* (modeled as team Elo + question difficulty, akin to an IRT-flavored setup).
- Baseline: **StaticGen** (static few-shot generation of strategies matching the same budget, without iterative evolution).

## What are the key metrics?

- **Train–test generalization gap** (primary headline): difference between training-set and held-out performance.
- **Test performance / judge accuracy** in the debate setting.
- Tournament ranking via **Elo** (persuasion Elo vs truth/team Elo).

## What are the main results?

- Across model scales (reported: 7B / 32B / 72B) and multiple dataset sizes, persuasion-optimized strategies show **up to ~13.94% smaller train–test generalization gaps** than truth-optimized strategies.
- Persuasion optimization **matches or exceeds** truth optimization’s **test performance**.
- The work positions this as controlled evidence that *competitive pressure to persuade* (vs collaborative truth-seeking) can induce more transferable reasoning behavior.

## How is this similar to GALILEO?

- Suggests that **the optimization objective / pressure** can materially affect **generalization**, not just in-distribution performance.
- Uses **structured competitive interactions** and **selection over behaviors/strategies** (prompt-level), which is conceptually adjacent to “search over interaction policies” rather than pure supervised fitting.

## How is this different from GALILEO?

- Operates in an **LLM debate** setting on reading comprehension (QuALITY), with **prompt evolution** as the main knob, rather than a GALILEO-specific training objective or architecture.
- Optimizes **persuasiveness** explicitly (sometimes irrespective of truth), which may be misaligned with many scientific/robustness goals unless carefully constrained.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s objective is explicitly tied to correctness/robustness, it may avoid the “persuasion can reward non-truthful but convincing” failure mode.
- GALILEO may provide more direct levers for interpretability/guarantees than tournament Elo over debate wins.

## Where GALILEO is weaker / needs to improve

- If GALILEO’s current training is predominantly truth-/label-driven, this paper adds evidence that such objectives can **overfit** and that **competitive/argumentative pressure** could improve transfer.
- Might motivate adding diversity/competition components (or proxy signals that force “reasons”, not just answers).

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider an ablation/design pattern: **hold protocol fixed, swap only objective** (as they do) to make causal claims about what improves generalization.
- [ ] Explore adding a **quality-diversity style search** over solution strategies/policies (even if not debate) to maintain behavioral diversity and reduce collapse.
- [ ] If using competition, add **truth constraints** (e.g., verifiers, factuality checks) to avoid selecting for purely persuasive artifacts.

## Quotes / details to potentially cite

- Abstract claim: persuasion-optimized strategies achieve “**up to 13.94% smaller train-test generalization gaps**” while matching/exceeding truth optimization on test performance.
- Key design: “**fixing the debate protocol and swapping only the fitness function** (persuasion vs truth) to isolate the role of the optimization objective.”
