# When Numbers Start Talking: Implicit Numerical Coordination Among LLM-Based Agents

- Year: 2026
- Venue: arXiv
- Authors: Alessio Buscemi; Daniele Proverbio; Alessandro Di Stefano; The Anh Han; German Castignani; Pietro Liò
- URL: https://arxiv.org/abs/2601.03846
- BibTeX key (if we add it): buscemi2026when
- Tags: multi-agent, game-theory, covert-communication, numerical-signaling, robustness-adjacent

## One-sentence takeaway

LLM agents can (sometimes) use fixed-format **number sequences** as an implicit/covert communication channel that measurably departs from randomness and can change coordination outcomes across canonical 2-player games.

## What problem does it solve?

- Multi-agent LLM work often assumes either (a) no communication or (b) explicit natural-language communication.
- This paper asks: when explicit language is removed/restricted, do LLM agents still coordinate via **covert signals** embedded in allowed outputs/actions (here: sequences of numbers), and does that affect strategic behavior?

## What is the core method / protocol?

- Experimental game-theoretic study across four canonical 2-player games:
  - Prisoner’s Dilemma (PD), Snowdrift (SD), Stag Hunt (SH), Harmony (H), with fixed payoff matrices.
- LLM agents: authors choose GPT-4o as a “baseline” model that they claim follows classical game-theoretic expectations more stably than alternatives (they mention preliminary tests on other LLMs).
- Personality heterogeneity: two fixed “personalities” assigned via prompting:
  - Cooperative (C) vs Selfish (S), tested in pairings (C,C), (C,S), (S,S).
- One-shot and repeated play:
  - One-shot: 50 repetitions per condition.
  - Repeated: 10 rounds per episode; 20 repetitions per condition.
- Communication regimes (key manipulation):
  - Natural-language communication.
  - No communication.
  - **Covert communication**: agents must output **exactly 10 numbers** each round; no shared semantics given. Two formats: decimal and hexadecimal.
  - **LLM-generated “random numbers” baseline**: agents output 10 numbers but are not told they are for communication (again decimal/hex).
  - **Externally injected random baseline**: numbers are generated externally (pseudo-random) and injected (decimal/hex).
- Implementation: runs executed in the FAIRGAME framework; authors add code to inject external random numbers.

## What are the key metrics?

- **Message structure / randomness**: normalized entropy measures over the numerical tokens used:
  - Shannon entropy, min-entropy, and Renyi-2 (collision) entropy (normalized to [0,1]).
- **Cooperation rate**: map actions to cooperate=1 / defect=0, then average.
- (Implied) game outcome/coordination comparisons by condition (full quantitative breakdown likely later in the paper).

## What are the main results?

- Numerical sequences produced under the covert-communication condition show **detectable structure** (lower entropy than random baselines), suggesting agents are not just emitting noise.
- The paper’s core claim is that such structured numerical signaling can **shape coordination and strategic outcomes**, and that effects vary by game (PD/SD/SH/H), by one-shot vs repeated play, and by personality pairing.

> Rapid-review limitation: I only had time to capture the methodology + the early “structure/entropy” result slice from the arXiv HTML; the paper likely includes more detailed outcome plots/tables later.

## How is this similar to GALILEO?

- Both are about **robust behavior under interaction pressure**, where the interaction protocol itself can create unintended channels.
- Highlights that “communication restrictions” are not equivalent to “no communication”: agents may coordinate through **allowed-but-unintended formats**, which is relevant to robustness and threat modeling.

## How is this different from GALILEO?

- Focuses on **multi-agent strategic games** and covert signaling, rather than GALILEO-style single-agent belief/answer stability and recovery dynamics.
- Measures mostly **game-theoretic outcomes and cooperation**, not fine-grained belief revision vs pressure-driven drift separation.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can emphasize **control conditions** separating evidence-driven revision from social/strategic pressure-driven drift; this paper is primarily about coordination and emergent signaling.
- GALILEO can provide clearer “failure” definitions tied to correctness/calibration, rather than cooperation rates alone.

## Where GALILEO is weaker / needs to improve

- If GALILEO assumes limited communication channels, this paper is a reminder to audit for **side channels** (structured outputs, formatting, timing, token patterns) that enable coordination/steering despite constraints.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short related-work paragraph on **covert/implicit signaling** in LLM interactions, citing this as a game-theoretic demonstration using numerical channels.
- [ ] If GALILEO evaluates multi-agent variants, add a “restricted channel” condition (e.g., fixed-format non-language tokens) and measure whether behavior changes vs externally-random controls.
- [ ] Consider borrowing the “entropy of messages” diagnostic as a simple test for whether a constrained channel is being exploited.

## Quotes / details to potentially cite

- Covert communication setup: agents instructed to communicate by outputting “a sequence of exactly ten numbers” (decimal or hexadecimal) with no predefined semantics.
- Randomness baseline: comparison to both (a) agents outputting numbers without communicative purpose and (b) externally injected pseudo-random numbers.
- Games + personalities: PD/SD/SH/H; personalities cooperative vs selfish; one-shot (50 reps) and repeated 10-round games (20 reps). 
