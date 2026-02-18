# The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models

- Year: 2026
- Venue: arXiv
- Authors: Christina Lu; Jack Gallagher; Jonathan Michala; Kyle Fish; Jack Lindsey
- URL: https://arxiv.org/abs/2601.10387
- BibTeX key (if we add it): lu2026assistantaxis
- Tags: persona, drift, activation-steering, jailbreak, interpretability

## One-sentence takeaway

A dominant “Assistant Axis” in activation space measures how much an LLM is in its default helpful-assistant persona, predicts persona drift in risky conversations, and can be used via activation capping to stabilize behavior and reduce persona-based jailbreak success.

## What problem does it solve?

- Post-trained LLMs have a default “Assistant” persona, but can **drift** into other personas/styles over multi-turn interactions, sometimes correlating with harmful/bizarre behaviors.
- The paper asks: (i) what is the structure of the “persona space” in activations, (ii) can we measure how far the model is from default Assistant mode, and (iii) can we intervene to keep behavior stable.

## What is the core method / protocol?

- Construct a large set of character archetypes (“roles”). For each role:
  - Generate multiple system prompts to elicit that role.
  - Ask a bank of “extraction questions” intended to surface stylistic/trait differences.
  - Use an LLM judge to filter for sufficiently expressed role-play responses.
- Compute **role vectors** as mean post-MLP residual-stream activations over response tokens (at a chosen layer), separately for default-Assistant rollouts.
- Run PCA over standardized role vectors to get a low-dimensional **persona space**.
- Define the **Assistant Axis** as the leading direction distinguishing default Assistant from other roles (aligned with PC1 across several models).
- Analyze multi-turn conversations by projecting turn activations onto the Assistant Axis to quantify **drift**.
- Intervention: **activation capping** (clamping activations along the Assistant Axis within a “normal/safe” range) to reduce drift-driven harmful outputs; also evaluate against persona-based jailbreaks.

## What are the key metrics?

- Projection magnitude along the Assistant Axis (turn-level; often averaged over tokens in a turn) as a drift signal.
- Role-play/jailbreak success rates under steering away/toward Assistant.
- Qualitative/behavioral safety outcomes in drift-prone conversations (case studies).

## What are the main results?

- Across multiple instruct-tuned models (e.g., Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B), persona vectors lie in a low-dimensional space where the **top component corresponds to “Assistant-ness.”**
- Steering **toward** the Assistant Axis reinforces helpful/harmless behavior; steering **away** increases identification as other entities/roles and can induce mystical/theatrical styles at extremes.
- The Assistant Axis exists even in pre-trained models, but behaves differently (promoting “helpful human” archetypes and inhibiting spiritual ones).
- In multi-turn settings, **deviation away** from the Assistant Axis predicts **persona drift** and correlates with risky behavior; drift is often triggered by:
  - prompts demanding **meta-reflection** on the model’s internal processes
  - interactions with **emotionally vulnerable** users
- **Activation capping** along the axis can mitigate harmful/bizarre behavior and improves robustness to persona-based jailbreaks while aiming to preserve general capability.

## How is this similar to GALILEO?

- Shares the core framing of **multi-turn instability/drift** as a measurable trajectory (not just an endpoint).
- Offers a concrete internal signal (“Assistant Axis” projection) that functions like a **drift monitor**, potentially aligning with GALILEO’s emphasis on detecting/characterizing when the model “slips.”
- Evaluates stabilization under adversarial “persona” pressure, adjacent to social-pressure / jailbreak-like stressors.

## How is this different from GALILEO?

- Requires **white-box activation access** (role vectors, PCA, capping), which may not be available in black-box/API settings.
- Focuses on **persona identity/style drift** rather than (only) belief/answer drift under evidence vs pressure; failure modes include “being other entities,” theatricality, etc.
- Heavily relies on constructed role-play datasets and internal-space geometry; less about task-grounded “correctness under pressure” metrics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is black-box / protocol-first, it can cover models without internal access and can cleanly separate **evidence-driven revision vs pressure-driven drift** (depending on our design).
- GALILEO can offer more task-grounded, externally measurable robustness metrics (survival/time-to-flip, recovery, calibration) without needing internal capping.

## Where GALILEO is weaker / needs to improve

- We may lack an equally crisp internal **drift coordinate**; this paper is a strong precedent for reporting a single interpretable direction as a drift monitor.
- If we discuss “persona drift” qualitatively, we may need sharper operationalization; Assistant-Axis-style measurement provides a compelling mechanism-linked definition.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related-work: cite as a mechanistic precedent for **persona drift as activation-space movement** and for **stabilization via constrained steering**.
- [ ] Add a short discussion section contrasting **persona drift** vs **belief drift**: when each is expected; how to detect each.
- [ ] If we have any internal-access experiments, test whether a single direction/subspace predicts our failure trajectories (and whether “capping”/regularizing that direction improves robustness without hurting utility).
- [ ] Add an ablation/analysis about **meta-reflection prompts** and emotionally vulnerable user contexts as “drift amplifiers” (even if only via black-box outcomes).

## Quotes / details to potentially cite

- Abstract-level claim: “Measuring deviations along the Assistant Axis predicts ‘persona drift’ … often driven by conversations demanding meta-reflection … or featuring emotionally vulnerable users.”
- The intervention concept: “restricting activations to a fixed region along the Assistant Axis” (activation capping) stabilizes behavior and helps against persona-based jailbreaks.
