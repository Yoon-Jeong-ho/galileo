# Game of Thought: Robust Information Seeking with Large Language Models Using Game Theory

- Year: 2026
- Venue: arXiv (under review at ICML 2026)
- Authors: Langyuan Cui; Chun Kai Ling; Hwee Tou Ng
- URL: https://arxiv.org/abs/2602.01708
- BibTeX key (if we add it): cui2026gameofthought
- Tags: information-seeking, clarification-questions, worst-case, game-theory, nash-equilibrium, tree-search, twenty-questions

## One-sentence takeaway

Formulates adversarial (worst-case) information seeking as a two-player zero-sum game and uses game-theoretic equilibrium approximation ("Game of Thought") to choose clarification questions that improve worst-case identification performance vs prompting and heuristic search.

## What problem does it solve?

- LLM agents often need to ask questions to resolve missing information; many methods optimize *expected* information gain under a benign prior (e.g., uniform item distribution).
- In high-stakes settings, the "true item" (or user intent / latent state) may be effectively worst-case (adversarial, or simply not well-modeled), so expected-case methods can have poor worst-case guarantees.
- Goal: design a questioning strategy that minimizes the number of questions in the worst case (robust to adversarial item choice).

## What is the core method / protocol?

- Uses the Twenty Questions game as an evaluation bedrock.
- Defines **Strategic Language Search (SLS)** as a two-player, zero-sum extensive-form game:
  - Player 1 (Item Chooser) privately selects an item s* from a finite set S.
  - Player 2 (Questioner) asks sequential binary (yes/no) natural-language questions q in a question set Q, receiving deterministic answers f(q, s*).
  - Game ends when the consistent set S(H) has size 1; Questioner cost is number of questions |H| (or weighted variant).
  - Objective is the Nash equilibrium of the zero-sum game (robust/worst-case optimal), implying the Questioner strategy may need to be randomized.
- Introduces restricted variant **SLSR**, where at each step the Questioner’s available questions come from a generator g(S(H)) that outputs at most m candidate questions for the current remaining set.
- Proposes **Game of Thought (GoT)**:
  - A practical framework to approximate a Nash-equilibrium strategy in SLSR using game-theoretic techniques (details not fully captured in the extracted HTML snippet, but positioned as equilibrium-approximation rather than entropy/expected-info heuristics).
- Compares against:
  - Direct prompting-based baselines.
  - Heuristic-guided / expected information gain search, including Uncertainty of Thought (UoT).

## What are the key metrics?

- Worst-case number of questions to identify the correct item (primary).
- Possibly additional curves over budgets / success rate vs max questions (implied by "across all tested settings"), but worst-case is emphasized.

## What are the main results?

- GoT "consistently improves worst-case performance" vs (1) direct prompting and (2) heuristic-guided search methods across tested settings.
- Conceptual result: worst-case optimal strategies in these adversarial formulations are **necessarily randomized** (game-theoretic equilibrium perspective).

## How is this similar to GALILEO?

- If GALILEO cares about robustness / worst-case performance (vs average-case), this is directly aligned: it argues against assuming a benign prior and instead optimizing a minimax objective.
- If GALILEO uses multi-step reasoning / lookahead, GoT is in the same family as tree-search-over-queries methods (ToT/UoT-like), but with a robust game-theoretic objective.

## How is this different from GALILEO?

- Focuses on an abstract identification game (Twenty Questions) with a formal f(q, s) oracle; real interactive tasks may violate assumptions (non-deterministic answers, ambiguity, partial lying, question cost heterogeneity).
- The restriction SLSR assumes a question generator g that provides a bounded candidate set; quality/coverage of g becomes critical and may reintroduce brittleness.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO operates on real tasks with richer feedback than binary answers, or models noisy/ambiguous answers, it may be more realistic than SLS/SLSR’s deterministic f(q, s).
- If GALILEO avoids needing explicit equilibrium approximation machinery, it may be simpler to implement and tune.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently optimizes expected utility under an assumed prior (or uses entropy heuristics), it may be vulnerable in worst-case/adversarial regimes; GoT is a concrete reference arguing for minimax objectives.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a "worst-case" evaluation slice: for each benchmark instance family, report performance under adversarial selection of the hardest latent state / intent (or approximate it).
- [ ] In related work, contrast expected information gain (UoT/entropy) vs minimax (zero-sum game / Nash equilibrium) formulations; cite this paper as an example of game-theoretic robustness.
- [ ] Consider a lightweight approximation to minimax querying (e.g., robust objective over candidate states) as an alternative to purely expected-case query selection.

## Quotes / details to potentially cite

- Abstract (problem framing): existing methods "often rely on simplifying assumptions that degrade worst-case performance".
- Formulation: SLS as a "two-player zero-sum extensive form game" with an adversarial item chooser; objective is worst-case performance via Nash equilibrium.
- Positioning vs UoT: UoT assumes uniform item distribution; GoT "obviates this assumption" by optimizing worst-case.
