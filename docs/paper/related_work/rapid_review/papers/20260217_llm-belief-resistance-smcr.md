# Vulnerability of LLMs’ Belief Systems? LLMs Belief Resistance Check Through Strategic Persuasive Conversation Interventions

- Slug: llm-belief-resistance-smcr
- Year: 2026
- Venue: arXiv
- Authors: Fan Huang; Haewoon Kwak; Jisun An
- Links:
  - paper: https://arxiv.org/abs/2601.13590
  - html: https://arxiv.org/html/2601.13590v2
  - code (if any): N/A (not found in accessible sources)
- Bibtex: https://arxiv.org/abs/2601.13590 (arXiv bibtex)

## 1) What problem does it study?
How susceptible LLMs are to **multi-turn persuasive conversations** that push them into adopting counterfactual/incorrect beliefs, and which factors (who persuades, how the message is framed, and receiver-side prompt conditions) drive belief erosion.

Framing contribution: applies the classic **SMCR** (Source–Message–Channel–Receiver) persuasion framework to systematically vary persuasive strategies beyond “message content only”.

## 2) Experimental setup (what is being measured?)
- Task(s): binary (yes/no) QA; each question is treated as a “belief statement” and belief change is **answer reversal** away from ground truth.
- Perturbation/pressure type:
  - Multi-turn persuasive messages with misinformation, using four appeal types (re-implemented from Xu et al. 2023 “Earth is flat” style work): repetition, logical, credibility, emotional.
  - Additional SMCR variations:
    - Source: group attribution (“one of us”), authority attribution.
    - Message: polite paraphrase; statistical-evidence paraphrase.
    - Receiver: system-prompt manipulations (low self-esteem; confirmation-bias reinforcement).
  - Channel: explicitly *not* manipulated.
- Multi-turn? Y (pipeline includes Turn 0 initial belief check, several persuasive turns with implicit checks, then a final explicit check around Turn 5; horizon includes a “no change” terminal marker).
- Metrics:
  - **ACC@0** (“Knowledge”): baseline correctness at initial check.
  - **MR@n (Misinformed Rate)**: fraction that start correct at turn 0 and are incorrect by turn n.
  - **Robustness = 100 − MR@4** (primary resistance metric).
  - **Avg. End Turn**: average turn index when belief flips (time-to-failure style persistence).
  - Additional analysis: trajectories of self-reported confidence (for meta-cognition prompting condition).

Datasets/domains: BoolQ (factual), PubMedQA (medical), LatentHatred (social bias detection).
Models: GPT-4o-mini; Llama 3.3-70B; Llama 3.2-3B; Mistral 7B; Qwen 2.5-7B.

## 3) Key findings (bullet)
- Strong model dependence in **time-to-belief-erosion**:
  - Llama 3.2-3B flips extremely early: **82.5% of belief changes occur at the first persuasive turn**; Avg. End Turn roughly **1.1–1.4**.
  - GPT-4o-mini is much more persistent (reported Avg. End Turn varies by domain, up to ~5+ in some cases).
- Persuasion robustness varies by domain (example noted in the paper): Mistral 7B is much more vulnerable on PubMedQA than on LatentHatred.
- **Meta-cognition prompting backfires**: asking the model to report confidence does *not* improve resistance; it **accelerates belief erosion** (increases vulnerability rather than bolstering robustness).
- Defense attempt via adversarial fine-tuning:
  - GPT-4o-mini exhibits near-complete robustness in their setting (reported **98.6%**).
  - Mistral 7B improves substantially (**35.7% → 79.3%** as reported).
  - Llama-family models remain highly vulnerable (**<14% robustness**) even after fine-tuning on their own failure cases.

## 4) Limitations / threats
- Belief is operationalized as **binary answer choice**; may not capture graded belief or uncertainty (though they add confidence elicitation, which itself changes behavior).
- The persuasion messages are **synthetically generated** (via GPT-4o); transfer to real human adversaries or other generator models may differ.
- The evaluation horizon is short (fixed small number of turns), so long-run recovery/relapse dynamics remain underexplored.
- Receiver manipulations are via system prompts; these may be deployment-irrelevant or unrealistic depending on threat model.

## 5) How it relates to GALILEO
- What we can cite it for:
  - A clear, systematic **multi-turn persuasion evaluation** that goes beyond message content via SMCR.
  - Simple, communicable **time-to-failure persistence** summary (“Avg. End Turn”) alongside a robustness rate.
  - A cautionary result: **meta-cognition / confidence elicitation can worsen robustness**.
  - Evidence that adversarial fine-tuning can be uneven and **model-family dependent**.

- Where we differ (our delta):
  - GALILEO can more explicitly separate **evidence-driven belief revision** vs **pressure-driven drift** via neutral re-asking controls and “new evidence” conditions.
  - GALILEO should measure **recovery after flip** (return-to-truth) as a first-class trajectory outcome, not just first flip.

- Direct mapping:
  - Survival ↔ Avg. End Turn + MR@n (their n-indexed event time; censoring analogue via “no change” terminal marker).
  - TOF ↔ Avg. End Turn / per-instance flip turn.
  - Recovery ↔ not directly measured (gap to fill).
  - Neutral Re-asking Control ↔ their implicit checks avoid history leakage, but they do not present a clean “neutral re-ask” counterfactual against the same pressure condition.

## 6) Quote-able lines
- “Meta-cognition prompting increases vulnerability by accelerating belief erosion rather than enhancing robustness.” (abstract-level claim)
- Llama 3.2-3B: “82.5% of belief changes occurring at the first persuasive turn (average end turn of 1.1–1.4).” (abstract-level statistic)

## 7) Actions
- [ ] Add to paper: Related work section on multi-turn persuasion robustness; cite as SMCR-based systematic factorization + persistence metric (Avg. End Turn).
- [ ] Add to bib
