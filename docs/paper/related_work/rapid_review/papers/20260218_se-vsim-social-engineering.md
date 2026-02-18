# Personalized Attacks of Social Engineering in Multi-turn Conversations: LLM Agents for Simulation and Detection

- Year: 2025
- Venue: arXiv
- Authors: Tharindu Kumarage, Cameron Johnson, Jadie Adams, Lin Ai, Matthias Kirchner, Anthony Hoogs, Joshua Garland, Julia Hirschberg, Arslan Basharat, Huan Liu
- URL: https://arxiv.org/abs/2503.15552
- BibTeX key (if we add it): kumarage2025sevsim
- Tags: social-engineering, multi-turn, persuasion, detection, agents, robustness

## One-sentence takeaway

Uses an LLM-agent “attacker vs victim” simulator grounded in social-engineering mechanisms + Big-Five victim traits to generate multi-turn manipulation dialogues, then shows personality/strategy-aware, delegated detection outperforms naive “look for sensitive info” detectors.

## What problem does it solve?

- Multi-turn chat-based social engineering (CSE) is harder than single-turn phishing detection because attacks evolve: trust-building and strategy shifts can happen across turns, often before any explicit sensitive-info disclosure.
- Existing detectors (including LLM-prompted ones) over-rely on spotting explicit sensitive information exchange, missing partially-successful attacks.
- Need (a) realistic-ish multi-turn data for studying the space and (b) detection approaches that incorporate conversation dynamics and victim-specific vulnerability factors.

## What is the core method / protocol?

- **SE-VSim**: dual-agent simulation framework.
  - Attacker agent is conditioned on a goal (role + intent): impersonate recruiter / funding agency / journalist; attempt to extract target information (PII, financial details, IP).
  - Victim agent is conditioned on **Big Five personality traits** (natural-language persona descriptions).
  - Conversation generation alternates attacker/victim for a fixed budget (reported as 10 turns), producing benign vs malicious conversations by including/removing malicious intent.
- **Dataset**: 1,350 simulated conversations (900 malicious, 450 benign) spanning roles/info-types/trait profiles.
- **Annotations**:
  - Attack success for malicious conversations on a 3-point scale (highly/partially/unsuccessful). Humans label; they also test an LLM (GPT-4o-mini) as an annotator and report high agreement (Fleiss’ kappa reported as ~0.796).
  - Attack strategy taxonomy labels via LLM judge (multi-label; too expensive to do manually).
- **SE-OmniGuard** (proof of concept detector): “delegate” architecture.
  - Control agent (larger LLM) orchestrates worker agents (smaller LLMs) that separately assess victim traits, attacker strategies, and info exchange, then the controller synthesizes.
  - Evaluated against prompted LLM baselines + a prior pipeline (ConvoSentinel, with some component substitutions).

## What are the key metrics?

- Detection: accuracy and F1 for identifying malicious SE attempts (with analysis by success level: unsuccessful vs partially successful vs successful).
- Annotation reliability: inter-annotator agreement (Fleiss’ kappa) for success labels.

## What are the main results?

- Baseline detectors (prompted LLMs, ConvoSentinel-style pipeline) perform poorly, especially on **partially successful** attacks where there is trust-building but not yet direct disclosure.
- SE-OmniGuard’s delegated, personality/strategy-aware analysis improves accuracy over baselines (paper emphasizes the gap is driven by baseline over-focus on sensitive info exchange).
- The simulation analysis suggests correlations between certain Big-Five traits (notably conscientiousness and agreeableness) and attack success patterns, aligning qualitatively with prior SE literature.

## How is this similar to GALILEO?

- Same broad theme: **multi-turn robustness under adversarial social pressure**, where failures are trajectory-level (trust-building, conformity/compliance, gradual manipulation), not just single-turn mistakes.
- Reinforces the need for **turn-by-turn state tracking** (strategy, vulnerability exploitation) and evaluation slices like “partial success” vs “full success,” analogous to “drift before flip” in sycophancy/pressure settings.

## How is this different from GALILEO?

- Domain focus is **social engineering / cybersecurity** rather than general belief revision/sycophancy robustness; success criteria include information extraction and trust establishment.
- The dataset is primarily **LLM-simulated** conversations (with some human validation/labeling), which may not match real user dynamics.
- Defense is framed as an **external detector/guard** rather than an intrinsic model-side robustness method (though the delegate pattern is conceptually adjacent to multi-module evaluators).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO emphasizes controlled multi-turn protocols and “belief/answer consistency under pressure,” it may offer a cleaner experimental lens than SE-VSim’s scenario-specific goals (recruiter/funder/journalist) and simulated dialogue artifacts.
- GALILEO likely targets more generalizable failure modes (sycophancy/belief drift) rather than SE-specific indicators.

## Where GALILEO is weaker / needs to improve

- This paper highlights a concrete evaluation axis GALILEO should mirror: **detecting early-stage/partial manipulation** before any overt “bad event” occurs.
- Personality-/user-profile-conditioned susceptibility analysis is an angle GALILEO may not currently cover (even if the “victim traits” are simplistic).
- The delegate/ensemble decomposition suggests robustness may benefit from **modular critiques** (strategy detector + vulnerability detector + info-exchange detector), which GALILEO could adapt for evaluation or interventions.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an evaluation slice akin to **“partially successful manipulation”**: quantify how often a model begins to comply/align (or change stance) *before* an explicit contradiction or final flip.
- [ ] Consider annotating multi-turn trajectories with **attack-strategy labels** (even coarse) and evaluate robustness by strategy category (authority, urgency, reciprocity, etc.).
- [ ] In related-work text, cite this as evidence that multi-turn adversarial settings require more than “detect sensitive info” / single-turn checks; robust methods need trajectory-level modeling.
- [ ] Consider whether GALILEO could borrow the **delegate pattern** as an evaluation harness: multiple specialized judges producing signals that are combined into a stability/robustness score.

## Quotes / details to potentially cite

- “Detecting SE in multi-turn, chat-based interactions is especially challenging due to the dynamic nature of these conversations, where the interaction evolves with each exchange.” (Intro)
- Dataset size and split: “1,350 simulated conversations” with “900 malicious” and “450 benign” (Intro/Conversation statistics).
- Attack success labeling: 3-level success metric; human labeling plus LLM annotator agreement (Fleiss’ kappa reported ~0.796).
- Key limitation of baselines: over-reliance on “sensitive information exchanges” misses partially-successful trust-building attacks.
