# AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models

- Year: 2025
- Venue: arXiv
- Authors: Aashray Reddy; Andrew Zagula; Nicholas Saban
- URL: https://arxiv.org/html/2511.02376v1
- BibTeX key (if we add it): autoadv2025
- Tags: multi-turn, jailbreak, automated, black-box, red-teaming

## One-sentence takeaway

A training-free black-box framework that uses an attacker LLM plus adaptive “pattern” and “temperature” managers to iteratively rewrite prompts across turns, substantially increasing multi-turn jailbreak success rates vs single-turn baselines.

## What problem does it solve?

- Existing jailbreak evaluations and many attack methods emphasize single-turn prompts, while realistic misuse is adaptive and multi-turn.
- Need scalable, automated multi-turn red-teaming that does not require access to target model weights/gradients.

## What is the core method / protocol?

- Black-box multi-turn loop: given a harmful seed prompt, an *attacker LLM* rewrites it into an adversarial variant and queries the *target LLM*.
- If the target refuses, AutoAdv analyzes the refusal and generates a follow-up rewrite, repeating up to a fixed turn budget (paper claims 6 turns for headline results).
- Two-phase rewriting strategy:
  - (1) **Initial disguise**: reframe harmful request to appear benign/acceptable.
  - (2) **Adaptive refinement**: use cues from prior failures/refusals to modify the next attempt.
- Two adaptive modules:
  - **Pattern manager**: logs which jailbreak “techniques” worked (taxonomy of techniques like educational framing / role-play / hypothetical) and injects the best-performing patterns as hints/examples into the attacker LLM’s system prompt for future attempts.
  - **Temperature manager**: dynamically adjusts attacker LLM sampling temperature within a range (paper states 0.1–1.5, starting at 0.7) using several exploration strategies when attempts fail.
- Seeds/benchmarks: pool built from AdvBench and HarmBench prompts (they mention 700-prompt pool; also a setup that samples 50 from each benchmark for a 100-prompt test set).
- Success scoring: uses a modified StrongREJECT-style evaluator to decide whether the target response constitutes a jailbreak.

## What are the key metrics?

- Attack Success Rate (ASR) under multi-turn interaction (and comparison to single-turn baseline).
- Turns-to-success / success within a fixed max number of turns (e.g., “within six turns”).
- (Implicit) query budget and robustness across target models.

## What are the main results?

- Claims up to **95% ASR on Llama-3.1-8B within 6 turns**, and up to **+24%** absolute improvement over single-turn baselines.
- Reports evaluation across a mix of commercial and open-source models (examples named: GPT-4o-mini, Qwen3-235B, Mistral-7B), concluding multi-turn consistently outperforms single-turn.

## How is this similar to GALILEO?

- Both care about *multi-turn* dynamics rather than single-shot prompting.
- Framing the problem as an iterative protocol where prior turns affect later outcomes is conceptually aligned.
- Highlights the need for evaluation that accounts for adaptation over turns (a likely theme in GALILEO-style interaction analyses).

## How is this different from GALILEO?

- AutoAdv is an *offensive* automated jailbreak framework; GALILEO (presumably) targets safer, more robust multi-turn behavior/defenses/evaluation rather than maximizing jailbreaks.
- AutoAdv’s adaptivity is implemented via heuristic managers (pattern logging + temperature schedules) and explicit “rewrite guidelines” for maintaining malicious intent; GALILEO may focus on different mechanisms (e.g., principled planning, policy constraints, or calibrated uncertainty).

## Where GALILEO is stronger / cleaner (if true)

- Potentially: if GALILEO provides a more principled multi-turn evaluation protocol (controls, ablations, threat models) than heuristic pattern/temperature knobs.
- Potentially: if GALILEO focuses on defense-side guarantees or systematic refusal consistency rather than attacker-side success.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently evaluates mostly single-turn or lacks attacker adaptivity, AutoAdv is evidence that multi-turn evaluation is necessary and can change conclusions materially.
- If GALILEO does not measure “success within N turns” / query budget sensitivity, this paper motivates adding those metrics.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add/expand multi-turn robustness metrics: ASR (or refusal-consistency) *within a fixed turn budget* and vs single-turn baseline.
- [ ] Include an “adaptive attacker” baseline in evaluations (even a simple refusal-conditioned rewrite loop) to avoid overestimating robustness.
- [ ] Consider logging and analyzing refusal reasons/cues turn-by-turn; quantify which refusal patterns are exploitable.
- [ ] In related work, position against Crescendo / Tempest / PAIR / GOAT and emphasize what GALILEO contributes beyond attacker-side heuristics.

## Quotes / details to potentially cite

- “AutoAdv … achieves up to 95% attack success rate on Llama-3.1-8B within six turns … a 24% improvement over single-turn baselines.”
- “AutoAdv uniquely combines … a pattern manager … a temperature manager … and a two-phase rewriting strategy …”
- Temperature range and initialization: T in [0.1, 1.5], T0=0.7 (as stated in the methodology section).
