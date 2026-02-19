# Must Read: A Systematic Survey of Computational Persuasion

- Year: 2025
- Venue: arXiv (cs.CL/cs.AI/cs.CY) survey
- Authors: Nimet Beyza Bozdag; Shuhaib Mehri; Xiaocheng Yang; Hyeonjeong Ha; Zirui Cheng; Esin Durmus; Jiaxuan You; Heng Ji; Gokhan Tur; Dilek Hakkani-Tur
- URL: https://arxiv.org/abs/2505.07775
- BibTeX key (if we add it): Bozdag2025ComputationalPersuasionSurvey
- Tags: persuasion, survey, safety, evaluation, manipulation

## One-sentence takeaway

A broad survey that frames "computational persuasion" around (i) AI as persuader, (ii) AI as persuadee (susceptibility/manipulation), and (iii) AI as persuasion judge (evaluation/detection/ethics), highlighting evaluation and safety as core open problems.

## What problem does it solve?

- Organizes a fast-growing, fragmented literature on AI-driven persuasion into a single taxonomy and agenda.
- Surfaces the dual-use/safety risk: LLMs can persuade humans, but can also be persuaded/manipulated (jailbreaks, bias reinforcement, adversarial prompting).
- Clarifies evaluation challenges: persuasiveness is subjective, context-dependent, and hard to measure robustly.

## What is the core method / protocol?

- Survey + taxonomy rather than a new algorithm.
- Key organizing lens (Fig. 1 in the paper):
  - **AI as Persuader**: generation of persuasive content and applications.
  - **AI as Persuadee**: influence/manipulation of AI systems (prompt attacks, agent-agent influence).
  - **AI as Persuasion Judge**: automated assessment of persuasion/manipulation/ethics.
- Anchors discussion in classic rhetoric (ethos/logos/pathos) and persuasion principles (e.g., Cialdini) while focusing on modern LLM capabilities and risks.

## What are the key metrics?

- Not a single metric (survey).
- Discusses/points to common evaluation targets that matter for computational persuasion:
  - **Persuasion effectiveness**: opinion/behavior change, agreement shift, preference change.
  - **Argument quality/strength** and strategy identification.
  - **Safety/ethics**: manipulation detection, policy compliance, harm risk.
  - **Robustness**: susceptibility to adversarial persuasive prompts; jailbreak success rates.

## What are the main results?

- Consolidates evidence that LLM persuasion can be strong and that multi-turn setups can increase effectiveness.
- Highlights that LLMs can be **both** powerful persuaders and vulnerable persuadees, which becomes more acute in multi-agent settings.
- Identifies open research directions: scalable evaluation, manipulation mitigation, responsible deployment of persuasive systems.

## How is this similar to GALILEO?

- If GALILEO involves agentic dialogue and user influence (recommendations, coaching, negotiation, alignment), this survey provides a useful framing and related-work map.
- The "judge" perspective aligns with building evaluators/critics to detect manipulation, measure persuasion, or enforce policy constraints.

## How is this different from GALILEO?

- This is a taxonomy/survey (no new system, no new benchmark, no implementation details).
- Focus is broader than any single task setting; includes social science grounding and safety/ethics discussion.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can contribute concrete, reproducible protocols (datasets, tasks, metrics) in contrast to survey-level synthesis.
- If GALILEO has a well-specified evaluation harness, it can operationalize the survey's calls for robust measurement.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently treats persuasion as a side-effect, this survey suggests making persuasion/susceptibility an explicit axis (including defenses against manipulative inputs).
- Need clarity on: what counts as "persuasion" vs. "helpful explanation"; and what safety constraints govern acceptable influence.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related-work writeup: structure the persuasion section explicitly as **persuader / persuadee / judge**.
- [ ] Add a "susceptibility" evaluation: measure how easily the system can be steered/manipulated into violating constraints or shifting policies.
- [ ] Define/justify a persuasion metric appropriate for GALILEO's domain (agreement shift, behavior intent, decision quality), and add guardrails for unethical manipulation.

## Quotes / details to potentially cite

- "[Computational persuasion] structured around three key perspectives: (1) AI as a Persuader ... (2) AI as a Persuadee ... and (3) AI as a Persuasion Judge ..." (abstract)
- Notes the subjectivity/context-dependence of persuasion and the resulting evaluation difficulty (abstract/introduction).
- Emphasizes that LLMs' persuasive ability and multi-turn interactions can increase persuasive effectiveness (introduction discussion).
