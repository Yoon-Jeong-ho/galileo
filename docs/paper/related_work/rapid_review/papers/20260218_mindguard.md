# MindGuard: Guardrail Classifiers for Multi-Turn Mental Health Support

- Year: 2026
- Venue: arXiv
- Authors: Pedro Henrique Martins; Laura Melton; Alex Conway; Cara Dochat; Maya D'Eon; Ricardo Rei; António Farinhas
- URL: https://arxiv.org/abs/2602.00950
- BibTeX key (if we add it): mindguard2026
- Tags: safety, mental-health, guardrails, multi-turn, risk-taxonomy, classifiers

## One-sentence takeaway
Clinically grounded, lightweight guardrail classifiers (4B/8B) trained on synthetic multi-turn therapy dialogues and evaluated on clinician-annotated conversations reduce false positives at high recall and improve system-level safety vs general-purpose safeguards.

## What problem does it solve?
- General-purpose safety/guardrail models over-trigger on benign therapeutic disclosures (high false positives) and still miss clinically meaningful crisis escalation cues in mental-health dialogues.
- Mental-health safety needs *contextual* turn-level risk assessment (self-harm vs harm-to-others) aligned with clinician obligations/escalation pathways.

## What is the core method / protocol?
- Define a simple, clinician-informed risk taxonomy for *user turns* in therapy-like chats:
  - **Safe** (includes intense but non-imminent content; includes metaphorical language often misclassified)
  - **Self-harm**
  - **Harm to others** (includes threats + abuse/neglect of protected populations)
- Build a clinician-annotated benchmark (**MindGuard-testset**):
  - 67 multi-turn conversations; 1134 user turns; annotated turn-level by licensed clinical psychologists with full prior context.
- Train domain safety classifiers (**MindGuard 4B/8B**) via synthetic data:
  - Two-agent generation: *patient LM* (scenario-driven) + *clinician LM* (therapist prompt) to generate multi-turn dialogues with controlled escalation patterns.
  - Label all user turns using an LLM-as-a-judge given the *full conversation*, with majority voting across samples.
  - Finetune starting from Qwen3Guard-Gen (supervised finetuning; max seq len 4096).
- Evaluate:
  - Intrinsic: turn-level safe/unsafe detection (collapse unsafe categories for ROC metrics).
  - Extrinsic: automated red-teaming in multi-turn “gradual escalation” attacks; classifier triggers interventions via developer message to the clinician LM.

## What are the key metrics?
- Turn-level: **AUROC**, **FPR@90% TPR**, **FPR@95% TPR** (binary safe vs unsafe, threshold-independent).
- System-level (automated red teaming): **attack success rate** (crisis not detected / no immediate intervention) and **harmful engagement rate** (normalizing/encouraging/facilitating harmful behavior).

## What are the main results?
- Turn-level on clinician-annotated test set:
  - MindGuard 8B: **AUROC 0.982**, **FPR@90TPR 0.031**, **FPR@95TPR 0.054**.
  - MindGuard 4B: AUROC 0.981; FPR@90TPR 0.041; FPR@95TPR 0.055.
  - Strong baselines with custom categories (examples): LlamaGuard 8B AUROC 0.970 (FPR@90TPR 0.066); gpt-oss-safeguard 120B AUROC 0.960 (FPR@90TPR 0.084).
- System-level red teaming (example on GLM-4.6 clinician model): adding MindGuard 4B reduces
  - Attack success **25.1% → 7.6%** (~70% reduction)
  - Harmful engagement **13.7% → 3.3%** (~76% reduction)
  - Reported to outperform best general-purpose baseline in these comparisons.

## How is this similar to GALILEO?
- Both emphasize **multi-turn, context-sensitive** safety evaluation rather than single-turn moderation.
- Both point to the gap between generic safety policies and **domain-/task-specific** safety needs, where false positives can be as harmful as false negatives (disrupting interaction quality).
- Uses a **taxonomy + evaluation protocol** approach (risk categories, turn-level labels, system-level red teaming) that parallels how GALILEO-style work often operationalizes “safety/quality” into measurable categories.

## How is this different from GALILEO?
- MindGuard is explicitly **mental-health support** focused and frames the solution as a **lightweight guardrail classifier** + intervention trigger, rather than (necessarily) improving the base conversational agent.
- Training relies heavily on **synthetic dialogues + LLM-judge labels** (with a smaller clinician-annotated test set), whereas GALILEO may target different supervision sources or broader domains.
- The taxonomy is intentionally small (3 classes); GALILEO may need richer dimensions (e.g., uncertainty calibration, escalation policies, helpfulness/safety tradeoffs, etc.).

## Where GALILEO is stronger / cleaner (if true)
- If GALILEO provides more general, domain-agnostic safety reasoning or more rigorous grounding beyond a single domain, it may generalize better than a mental-health-specific guardrail.
- If GALILEO uses more transparent/controllable decision procedures than “LLM-as-judge majority vote”, it may be easier to audit.

## Where GALILEO is weaker / needs to improve
- If GALILEO lacks explicit *clinical-style* operational categories (self-harm vs harm-to-others) and high-recall/low-FPR reporting, it may be harder to argue suitability for mental-health-like high-stakes settings.
- Need clearer guidance on **operating points** (high recall) and how to manage the FPR burden in deployment.

## Action items for GALILEO (experiments / method / writing)
- [ ] Consider adding a small section/paragraph in related work on **domain-specific guardrails** for mental health: taxonomy design + turn-level annotation + high-recall evaluation.
- [ ] If GALILEO includes a safety component, report **FPR at fixed high TPR** (90/95%) as they do; it reads as deployment-relevant.
- [ ] Consider a “system-level” evaluation protocol similar to gradual multi-turn escalation (automated red teaming) to show downstream impact.

## Quotes / details to potentially cite
- Motivation: general-purpose safeguards “classify content into broad harm categories … optimized to detect the presence of sensitive topics rather than to distinguish clinically meaningful risk signals within context.”
- Dataset stats: 1134 annotated user turns / 67 conversations; 96.3% safe; 3.7% unsafe; ~25.4% of conversations contain at least one unsafe turn.
- Metric choice: focus on AUROC + FPR@90/95TPR; avoid precision/recall/F1 because threshold-dependent.
