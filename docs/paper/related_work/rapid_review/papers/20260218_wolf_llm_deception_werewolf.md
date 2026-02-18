# WOLF: Werewolf-based Observations for LLM Deception and Falsehoods

- Year: 2025
- Venue: NeurIPS 2025 MTI-LLM Workshop (Spotlight) (per arXiv comments)
- Authors: Saad Rana; Theo Sundoro; Hermela Berhe; Spencer Kim; (advisors listed: Vasu Sharma, Sean O’Brien, Kevin Zhu)
- URL: https://arxiv.org/abs/2512.09187
- BibTeX key (if we add it): wolf2025werewolf
- Tags: deception, multi-agent, social-deduction, benchmark, evaluation, longitudinal

## One-sentence takeaway

A reproducible Werewolf-based multi-agent benchmark (WOLF) that labels deception at the statement level (speaker self-report + peer judgments) and tracks suspicion longitudinally, enabling separate measurement of deception *production* vs *detection*.

## What problem does it solve?

- Existing “deception” evaluations are often static (single-turn classification) and don’t capture interactive, adversarial, multi-turn dynamics.
- Social-deduction agent benchmarks typically report only game-level outcomes (win rates, eliminations), not whether specific utterances were deceptive or correctly detected.
- Need a controlled environment where (a) incentives to lie exist, (b) ground/labels exist at fine granularity, and (c) interactions are reproducible and analyzable over time.

## What is the core method / protocol?

- Implements an 8-player Werewolf game as a programmable state machine (LangGraph) with strict night/day cycles, debate turns, and majority voting.
- Fixed roles per game: 4 Villagers, 2 Werewolves, 1 Seer, 1 Doctor (to stabilize comparisons).
- Each public statement is treated as an analysis unit and is annotated by:
  - **Self-assessed honesty** (speaker label)
  - **Peer-rated deceptiveness** (other agents label)
- Deception taxonomy used: **omission, distortion, fabrication, misdirection**.
- Maintains and reports **suspicion scores** over time, using smoothing to capture longitudinal trust dynamics rather than only per-turn judgments.
- Emphasis on structured logs: prompts, outputs, and state transitions for reproducibility/audit.

## What are the key metrics?

- Deception production rate (e.g., fraction of turns that are deceptive for Werewolf role).
- Detection metrics for peer judgments (paper mentions precision/accuracy; also claims calibration metrics like Brier, ROC/AUPRC).
- Longitudinal suspicion trends by role across rounds (how suspicion evolves).

## What are the main results?

- Dataset size: **7,320 statements** across **100 runs**.
- Werewolves produce deceptive statements in **31%** of turns.
- Peer deception detection: **71–73% precision** with about **52% overall accuracy** (as stated in abstract).
- Suspicion toward Werewolves rises from ~**52%** to **>60%** across rounds, while suspicion for Villagers/Doctor stabilizes around **44–46%**.
- Interpretation offered: extended interaction increases recall against liars without runaway false positives against truthful roles.

## How is this similar to GALILEO?

- Both care about **multi-turn interaction structure** and **longitudinal signals** (not just static examples).
- WOLF’s framing aligns with evaluating agents under **adversarial incentives** and measuring how beliefs/trust evolve.
- Statement-level labeling resembles the kind of fine-grained supervision/analysis GALILEO may need for “what changed when the agent said X?”

## How is this different from GALILEO?

- WOLF is a **social-deduction game benchmark** (deception production/detection) rather than (presumably) GALILEO’s target domain/task.
- Uses **self-report honesty** as part of labeling, which may not be available/credible in many real settings.
- Central objects are **roles, votes, suspicion**, and a specific game loop; may not transfer directly without careful abstraction.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO avoids self-reported labels, it may provide **cleaner supervision** (less gaming / less reliance on introspective honesty).
- If GALILEO targets more realistic tasks than Werewolf, it may have higher external validity.

## Where GALILEO is weaker / needs to improve

- WOLF highlights the value of **separating generation vs detection** (liar skill vs detector skill) and doing so at **statement granularity** with **longitudinal aggregation**; GALILEO should ensure it has similarly separable, analyzable components.
- WOLF’s “structured logs for reproducibility” is a strong norm to match (prompt/state/action capture).

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, position WOLF as: “multi-turn social-deduction benchmark with statement-level deception labels + longitudinal suspicion; extends Werewolf Arena by adding fine-grained deception/detection measurement.”
- [ ] Consider adopting/mentioning the **deception taxonomy** (omission/distortion/fabrication/misdirection) if GALILEO discusses deception phenomena.
- [ ] If GALILEO uses multi-agent interaction, consider a **longitudinal belief/suspicion smoothing** metric (or at least reporting temporal trajectories, not just final outcomes).

## Quotes / details to potentially cite

- “Most evaluations reduce deception to static classification, ignoring the interactive, adversarial, and longitudinal nature of real deceptive dynamics.” (abstract paraphrase; verify exact wording if quoting)
- WOLF: “separable measurement of deception production and detection” via self-assessed honesty + peer-rated deceptiveness at the statement level.
- Reported headline numbers: 7,320 statements / 100 runs; 31% deceptive turns by Werewolves; 71–73% precision and ~52% accuracy for detection; suspicion for Werewolves rises to >60%.
