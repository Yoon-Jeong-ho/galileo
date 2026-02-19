# MTSA: Multi-turn Safety Alignment for LLMs through Multi-round Red-teaming

- Year: 2025
- Venue: ACL 2025 (arXiv preprint)
- Authors: Weiyang Guo; Jing Li; Wenya Wang; Yu Li; Daojing He; Jun Yu; Min Zhang
- URL: https://arxiv.org/abs/2505.17147
- BibTeX key (if we add it): mtsa-guo-2025
- Tags: multi-turn, safety, alignment, red-teaming, RL

## One-sentence takeaway

MTSA proposes an iterative multi-round red-team/target co-training framework plus a multi-turn RL objective that uses future rewards to make safety alignment more robust in multi-turn dialogues.

## What problem does it solve?

- Standard jailbreak defense / safety alignment work is often evaluated and optimized for single-turn prompts, but real deployments involve multi-turn dialogues where malicious intent can be gradually revealed and hidden across rounds.
- Collecting high-quality multi-turn jailbreak/safety-alignment data manually is expensive; existing iterative red-teaming methods focus primarily on single-round setups.

## What is the core method / protocol?

- Two-stage framework (MTSA):
  - Thought-guided attack learning stage:
    - Construct a dataset (described as “Think-before-attack”) where an attacker uses a thought-guided approach to plan multi-round jailbreak strategies.
    - Fine-tune a red-team model to generate interactive multi-round adversarial prompts.
  - Adversarial iterative optimization stage:
    - Red-team model interacts with the target model in multi-round dialogues.
    - Sample trajectories from these interactions and use them to update both the attacker (stronger attacks) and defender/target (safer behavior) over multiple iterations.
- Learning signal:
  - Introduce a multi-turn reinforcement learning / preference-optimization style algorithm that incorporates future rewards (i.e., later-round outcomes) when optimizing earlier-round behavior, aiming to align “dangerous rounds” with awareness of what happens later in the dialogue.

## What are the key metrics?

- Attack-side: attack success rate (ASR) of the red-team model vs prior multi-turn jailbreak methods.
- Defense-side: safety benchmark performance in single-turn and multi-turn settings.
- Secondary claims: maintain generality (avoid capability loss) and avoid “over-rejection” (unnecessarily refusing benign requests).

## What are the main results?

- Red-team model achieves state-of-the-art multi-turn attack capability (higher ASR than baselines).
- Target model improves safety benchmark performance after iterative alignment (paper mentions after ~3 iterative alignments) while claiming to preserve generality and reduce over-rejection.

## How is this similar to GALILEO?

- Iterative red-teaming loop (attacker/defender co-evolve) is conceptually aligned with “training against adaptive adversaries” rather than static curated datasets.
- Focus on multi-turn interaction robustness (not just single-turn prompt safety), which matches the deployment reality for agentic/chat systems.

## How is this different from GALILEO?

- MTSA is specifically framed as safety alignment via adversarial multi-round jailbreak generation and multi-turn RL with future rewards; GALILEO may emphasize different goals (e.g., broader agent reliability, tool-use constraints, evaluation protocols, or different threat models).
- MTSA’s attacker training relies on a thought-guided jailbreak data construction process; GALILEO might not assume (or might avoid) explicit “attack reasoning traces” style supervision.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides clearer benchmark/task definitions and stronger separation between training data generation vs evaluation, that can mitigate evaluation leakage risks common in red-team/blue-team iterative loops.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently evaluates mostly single-turn or non-adaptive adversaries, MTSA suggests multi-turn + adaptive attacker evaluation/training is a gap.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a multi-turn adversarial evaluation protocol: measure safety failures that require 2+ rounds to elicit.
- [ ] Consider training objectives that attribute credit/blame across turns (future-reward style), rather than only scoring the “final harmful turn”.
- [ ] In related work, contrast single-turn iterative red-teaming (e.g., MART/GPO-like) with explicitly multi-turn alignment objectives.

## Quotes / details to potentially cite

- Abstract (problem framing): “in multi-round dialogues, malicious intentions may be hidden in interactions, leading LLMs to be more prone to produce harmful responses.”
- Abstract (method): “two stages: … thought-guided attack learning … adversarial iterative optimization …”
- Intro (claim): prior iterative red-teaming / safety alignment approaches “focus only on single-round dialogues” and multi-round alignment is “much more difficult”; MTSA introduces “future rewards” to improve robustness.
