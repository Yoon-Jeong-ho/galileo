# What Helps Language Models Predict Human Beliefs: Demographics or Prior Stances?

- Year: 2025
- Venue: arXiv (cs.CL)
- Authors: Joseph Malone et al. (see arXiv submission)
- URL: https://arxiv.org/abs/2511.18616
- BibTeX key (if we add it): malone2025predictbeliefs (suggested)
- Tags: human-beliefs, social-reasoning, stance-prediction, demographics, prior-beliefs, ablations

## One-sentence takeaway

Off-the-shelf open-weight LLMs predict a person’s stance better when given either demographics or the person’s other stances, and best when given both—though which signal matters most varies by belief domain.

## What problem does it solve?

- Understand what information LLMs rely on when predicting human beliefs/stances (demographics vs prior stances vs both), and how that varies across domains.
- Provide evidence relevant to risks like stereotyping, privacy leakage, and personalized persuasion.

## What is the core method / protocol?

- Dataset: users and debates from an online debate platform (paper uses participants’ stance labels across topics).
- Task: predict an individual’s stance on a target issue.
- Conditions (ablations):
  - No context (blind baseline)
  - Demographics only
  - Prior beliefs only (the person’s stances on other issues)
  - Demographics + prior beliefs
- Models: “off-the-shelf open-weight LLMs” (not specified on the arXiv abstract page).
- Evaluation: compare stance prediction performance across conditions and across belief domains.

## What are the key metrics?

- Stance prediction accuracy / classification performance (exact metric details not in the abstract).
- Domain-wise breakdowns (importance of demographic vs prior-belief context varies by domain).

## What are the main results?

- Both demographics and prior-belief context improve over the blind baseline.
- Using both demographics + prior beliefs yields best performance “in most cases.”
- The relative marginal value of demographics vs prior stances varies substantially across belief domains.

## How is this similar to GALILEO?

- Shared theme: modeling and evaluating LLM behavior around human beliefs/stances (and the implications for persuasion/sycophancy/stereotyping).
- Uses controlled ablations of “social context” signals, which is conceptually similar to isolating drivers of model behavior.

## How is this different from GALILEO?

- Focuses on one-shot stance prediction, not multi-turn interaction dynamics, robustness under pressure, or belief revision over rounds.
- Emphasizes demographic vs correlational belief-structure signals; not primarily about conversational steering or stability/drift controls.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO evaluates multi-turn settings, it can better separate transient conversational effects (prompting/pressure) from stable “profile inference.”
- GALILEO can frame stronger causal-style tests for drift/revision (if it includes explicit controls), whereas stance prediction is more correlational.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not include explicit “demographics vs prior-stances” ablations, it may under-address a key pathway for stereotyping/privacy concerns.
- If GALILEO lacks domain-wise analyses, it may miss heterogeneous effects across topic domains.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an ablation axis for “available social context” (none vs demographics vs prior stances vs both) and measure downstream changes in behavior (e.g., agreement/sycophancy, persuasive susceptibility, belief drift).
- [ ] Report heterogeneity across topic domains (some domains may be driven more by demographics, others by belief correlations).
- [ ] Add a short related-work paragraph on privacy/stereotyping risks from stance prediction and profile inference.

## Quotes / details to potentially cite

- “We address these questions using data from an online debate platform, evaluating the ability of off-the-shelf open-weight LLMs to predict individuals' stance under four conditions: no context, demographics only, prior beliefs only, and both combined.”
- “We find that both types of information improve predictions over a blind baseline, with their combination yielding the best performance in most cases. However, the relative value of each varies substantially across belief domains.”
