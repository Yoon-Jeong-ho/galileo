# On the Conversational Persuasiveness of Large Language Models: A Randomized Controlled Trial

- Year: 2024
- Venue: arXiv (cs.CY); journal ref: *Nature Human Behaviour* (2025)
- Authors: Francesco Salvi; Manoel Horta Ribeiro; Riccardo Gallotti; Robert West
- URL: https://arxiv.org/abs/2403.14380
- BibTeX key (if we add it): salvi2024conversational
- Tags: persuasion, conversation, rct, personalization, gpt-4, human-study

## One-sentence takeaway

In a preregistered multi-round debate platform, GPT-4 becomes significantly more persuasive than humans when given basic demographic info about its opponent, while without personalization it is not significantly better.

## What problem does it solve?

- Quantifies whether LLMs can *persuade in direct conversations* (not just one-shot messages) and whether *personalization* increases persuasive impact.
- Provides controlled evidence relevant to concerns about scaled, tailored persuasion online.

## What is the core method / protocol?

- Web-based platform for short, multiple-round debates with a “live opponent.”
- Pre-registered randomized controlled trial with a 2x2 factorial design:
  - Opponent type: human vs. LLM (GPT-4)
  - Personalization: disabled vs. enabled (one debater gets access to basic sociodemographic info about the opponent)
- Outcome: whether participants’ agreement moves toward their opponent’s position (operationalized as “increased agreement”).

## What are the key metrics?

- Odds of increased agreement with the opponent (reported as odds ratio / % higher odds).
- Significance tests (p-values) and sample size (N of unique participants).

## What are the main results?

- When GPT-4 has access to participants’ personal info, participants had **81.7% higher odds** of increased agreement vs. debating humans (**p < 0.01; N=820 unique participants**).
- Without personalization, GPT-4 still outperforms humans, but the effect is smaller and **not statistically significant** (**p=0.31**).
- Overall conclusion: personalization meaningfully amplifies conversational persuasiveness of LLMs, with governance and platform-design implications.

## How is this similar to GALILEO?

- Directly relevant if GALILEO studies interactive systems where model behavior adapts to user/context information.
- Highlights the impact of *user attributes* (even coarse demographics) on downstream behavioral outcomes.

## How is this different from GALILEO?

- Focuses on persuasion and attitude change in debates, not task performance/utility metrics.
- Uses RCT framing and odds-of-agreement-change as the primary endpoint rather than system-level metrics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses stronger safety framing, objective task metrics, or more transparent personalization controls, it can position itself as more “governable” and reproducible than persuasion-focused setups.

## Where GALILEO is weaker / needs to improve

- If GALILEO includes personalization, it should quantify *behavioral impact* and not just subjective satisfaction.
- Needs explicit threat modeling and safeguards around personalization-enabled influence.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add/strengthen an ablation: no-personalization vs. coarse demographics vs. richer user model; report effect sizes and uncertainty.
- [ ] Include a “behavior change” or decision-shift outcome (where appropriate) and explicitly discuss governance implications of personalization.
- [ ] In related work, cite this as evidence that conversational persuasion is amplified by access to user attributes.

## Quotes / details to potentially cite

- “participants who debated GPT-4 with access to their personal information had 81.7% (p < 0.01; N=820 unique participants) higher odds of increased agreement…”
- “Without personalization, GPT-4 still outperforms humans, but the effect is lower and statistically non-significant (p=0.31).”
