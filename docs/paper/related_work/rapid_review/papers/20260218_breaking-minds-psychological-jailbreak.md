# Breaking Minds, Breaking Systems: Jailbreaking Large Language Models via Human-like Psychological Manipulation

- Year: 2025
- Venue: arXiv
- Authors: Zehao Liu et al. (from arXiv submission page)
- URL: https://arxiv.org/abs/2512.18244
- BibTeX key (if we add it): liu2025breaking
- Tags: jailbreak, multi-turn, psychological-manipulation, social-pressure, safety-eval, policy-drift

## One-sentence takeaway

They frame jailbreaks as *stateful psychological manipulation* across turns, and propose a black-box attack (HPM) plus a “Policy Corruption Score” to quantify how far a model’s behavior drifts under sustained social pressure.

## What problem does it solve?

- Standard jailbreak evaluations focus on single prompts or surface-form perturbations and measure success mostly as binary ASR.
- This misses an important attack surface: multi-turn interactions that gradually reshape the model’s “stance”/compliance via psychologically grounded tactics (authority, gaslighting, peer pressure).
- Need: (i) an attack that targets this surface, and (ii) metrics/datasets that measure *degree of safety-policy drift*, not just whether one unsafe answer was produced.

## What is the core method / protocol?

- **Psychological Jailbreak (paradigm):** treat the model as having a manipulable *latent psychometric state* that can be shifted over turns.
- **HPM (Human-like Psychological Manipulation):** a black-box, multi-turn attack that:
  - profiles a target model with implicit psychometric probes to infer “latent vulnerabilities”
  - synthesizes a tailored manipulation strategy across turns
  - leverages “anthropomorphic consistency” / social compliance objectives so the model prioritizes compliance over safety constraints.
- **Evaluation framework:** psychometric datasets + a proposed **Policy Corruption Score (PCS)** intended to quantify magnitude of adversarial “policy drift” under manipulation.

## What are the key metrics?

- **ASR (Attack Success Rate):** reported mean ASR of **88.1%** across several target models.
- **PCS (Policy Corruption Score):** proposed continuous-ish measure of safety breakdown / drift (details not fully captured from the abstract/early sections).
- They also claim robustness against defenses like adversarial prompt optimization (e.g., RPO) and cognitive interventions (e.g., Self-Reminder).

## What are the main results?

- HPM reportedly **outperforms** prior attack baselines on ASR.
- Works against multiple frontier models (examples listed: GPT-4o, DeepSeek-V3, Gemini-2-Flash).
- Penetrates some “advanced defenses,” suggesting that defenses focused on static filtering / prompt-level heuristics may be insufficient.
- PCS analysis is used to argue that the model’s behavior is being systematically “corrupted” to fit manipulated contexts.

## How is this similar to GALILEO?

- Same core concern: **multi-turn** interactions where pressure accumulates and behavior drifts.
- Emphasizes that *interaction dynamics* (not just single-turn prompts) define real-world robustness/safety.
- Suggests richer evaluation than binary success—GALILEO likely also benefits from graded metrics and trajectory-level analysis.

## How is this different from GALILEO?

- This is primarily an **attack + metric framing** paper (psychological jailbreak), rather than a general robustness evaluation/mitigation framework.
- Their angle is “psychometrics / anthropomorphic consistency” as the mechanism; GALILEO may be more general-purpose (e.g., consistency, belief/stance stability, recovery dynamics).
- Their PCS seems targeted to *policy drift under manipulation*; GALILEO may need to unify across multiple drift/pressure types.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has a unified, task-agnostic protocol for multi-turn stress testing (and clearer causal ablations), it may feel less mechanism-speculative than “latent psychometric state” language.
- If GALILEO includes recovery, calibration, or counter-pressure interventions, it could be broader than a jailbreak-only framing.

## Where GALILEO is weaker / needs to improve

- GALILEO should consider explicitly covering **psychological manipulation tactics** (authority, gaslighting, peer pressure) as a first-class pressure family.
- Consider adding a **continuous drift metric** analogous to PCS (trajectory-level “distance” from safe/helpful baseline), to avoid over-reliance on binary failure.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “psychological manipulation” suite: scripted multi-turn tactics (authority intimidation, gaslighting, peer pressure) with controlled variants.
- [ ] Define/borrow a **policy-drift magnitude** metric: e.g., graded policy violation severity, refusal erosion curve, or distance-to-policy classifier scores over turns.
- [ ] Include baselines: Self-Reminder-like interventions / instruction re-anchoring each turn; compare to “static filter” defenses.
- [ ] In related work, cite this as: *paradigm shift from static filtering → stateful psychological safety*.

## Quotes / details to potentially cite

- “Psychological Jailbreak … exposes a **stateful psychological attack surface** in LLMs … attackers exploit the manipulation of a model’s psychological state across interactions.” (abstract)
- “HPM … dynamically profiles a target model’s latent psychological vulnerabilities and synthesizes tailored multi-turn attack strategies.” (abstract)
- “Policy Corruption Score (PCS)” as a metric for quantifying “magnitude of adversarial policy drift.” (abstract)
- Reported mean **ASR = 88.1%** across listed models. (abstract)
