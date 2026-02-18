# Medical Large Language Model Benchmarks Should Prioritize Construct Validity

- Year: 2025
- Venue: arXiv (position paper; mentions ICML)
- Authors: Thomas Hartvigsen; Niloufar Golchini; Shiladitya Dutta; Frances Dean; Inioluwa Deborah Raji; Travis Zack
- URL: https://arxiv.org/abs/2503.10694
- BibTeX key (if we add it): hartvigsen2025constructvalidity
- Tags: evaluation, benchmarks, construct-validity, psychometrics, real-world-data

## One-sentence takeaway

Medical LLM leaderboard-style benchmark scores (often derived from licensing-exam questions) can fail to measure the real clinical capabilities they are used to claim, and should be empirically validated via a psychometrics-style construct-validity framework using real-world clinical data.

## What problem does it solve?

- Medical-LLM papers routinely make high-level capability claims (“reasons like a physician”, “clinical knowledge”) supported by benchmark performance.
- Many medical benchmarks are “arbitrarily constructed” (often from USMLE-style exam questions) and may not represent the real-world construct implied by those claims.
- Lack of a principled, quantitative framework for validating whether a benchmark actually measures the latent capability it purports to measure (construct validity), leading to misleading progress signals and inconsistent rankings across benchmarks.

## What is the core method / protocol?

- Conceptual move: treat LLM benchmarks like psychological tests; use psychometrics’ notion of **construct validity** (Cronbach & Meehl) to evaluate whether a benchmark measures an underlying construct.
- Argues benchmarks should be paired with explicit claims and then *empirically* validated against real-world clinical data (e.g., EHR-derived tasks/outcomes) by hospital systems.
- Includes proof-of-concept experiments using real-world clinical data to probe construct-validity gaps in popular medical LLM benchmarks (details not fully captured from the abstract/intro in this rapid pass).

## What are the key metrics?

- Not presented as a single metric in the abstract/intro; the emphasis is on *validity evidence* rather than accuracy alone.
- Implied evaluation axes (psychometrics-inspired): whether benchmark performance correlates with real-world task performance / outcomes for the claimed construct; whether it generalizes across settings; and whether benchmark-induced model rankings align with real-world utility.

## What are the main results?

- Reports “significant gaps” in construct validity for popular medical LLM benchmarks when tested with real-world clinical data (proof-of-concept).
- Advocates for a medical LLM evaluation ecosystem centered on creating **valid** benchmarks (explicit constructs + empirical validation), rather than leaderboard optimization.

## How is this similar to GALILEO?

- Shared thesis: **benchmark scores can be misaligned with the real capability we care about**.
- GALILEO’s focus (multi-turn robustness under pressure, belief revision vs drift, persuasion/sycophancy) also depends on task/benchmark design having high construct validity (i.e., that the protocol truly measures robustness vs artifacts like style, short-horizon cues, or prompt overfitting).
- Encourages moving from face-valid benchmarks to *empirically grounded* evaluation—aligned with GALILEO’s need to justify that its multi-turn protocols measure stability/robustness rather than superficial compliance.

## How is this different from GALILEO?

- Domain: medical LLM evaluation; uses clinical real-world data (EHR) as an anchoring substrate.
- Frame: explicitly imports psychometrics literature and “construct validity” terminology; GALILEO may currently use ML robustness framing more than formal validity theory.
- Focus is on benchmark validity rather than proposing a specific multi-turn adversarial/pressure-testing protocol.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides controlled, multi-turn stress tests with clear causal levers (pressure/persuasion, correction signals, repeated rounds), it may offer more *mechanistic* insight into failure modes than broad validity critiques.

## Where GALILEO is weaker / needs to improve

- Could benefit from adopting this paper’s explicit validity language: clearly stating the latent construct(s) (e.g., “robust belief revision under social pressure”) and collecting empirical validity evidence beyond internal benchmark performance.
- Need to articulate what “real-world” anchor corresponds to GALILEO constructs (e.g., human-judged consistency across rounds, downstream decision quality, resistance to socially-engineered instruction drift).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short “construct validity” subsection in the evaluation/method section: define constructs, threats to validity, and what evidence GALILEO provides.
- [ ] Consider validity-style checks: do GALILEO scores predict behavior on an external multi-turn setting (different prompts, domains, or naturally occurring conversation logs) meant to represent the same construct?
- [ ] When presenting leaderboard comparisons, add a cautionary note and (if possible) show correlation with an external criterion (human rating, downstream utility, or independently collected robustness outcomes).

## Quotes / details to potentially cite

- “Medical LLM benchmarks, much like those in other fields, are arbitrarily constructed using medical licensing exam questions.”
- Construct validity: “the ability of a test to measure an underlying ‘construct’, that is the actual conceptual target of evaluation.”
- “We… use real-world clinical data in proof-of-concept experiments to evaluate popular medical LLM benchmarks and report significant gaps in their construct validity.”
