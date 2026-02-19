# Dialogues with AI Reduce Beliefs in Misinformation but Build No Lasting Discernment Skills

- Year: 2025
- Venue: arXiv (cs.HC)
- Authors: Valdemar Danry; Paul Pu Liang; Andrew B. Lippman; Pattie Maes
- URL: https://arxiv.org/abs/2510.01537
- BibTeX key (if we add it): danry2025dialogues
- Tags: persuasion, misinformation, dialogue, human-study, overreliance

## One-sentence takeaway

AI-assisted dialogues can substantially improve people’s *in-the-moment* misinformation judgments, but over repeated use may *reduce* users’ unaided discernment on new items (dependency / overreliance effect).

## What problem does it solve?

- Understand whether conversational AI fact-checking/help *teaches* users transferable misinformation-detection skills, vs. merely helping them on the assisted instances.
- Quantify longitudinal effects of repeated AI assistance on unaided performance.

## What is the core method / protocol?

- Longitudinal (month-long) user study.
- Task: participants classify news headline–image pairs as real vs. fake.
- Protocol includes:
  - an evaluation phase without AI assistance,
  - an AI-assisted phase where participants “discussed their assessments with an AI system”,
  - followed by an unassisted evaluation on *unseen* items.
- Reported sample size: 67 participants.

## What are the key metrics?

- Accuracy on real/fake classification:
  - During AI-assisted sessions.
  - Unassisted performance on new/unseen items over time (week-by-week / end-of-study).
- (Implied) belief change / confidence could be tracked, but the headline results emphasized accuracy shifts.

## What are the main results?

- AI assistance yields large immediate gains during assisted sessions (reported ~+21% average improvement).
- Unassisted performance on new items *declines* over time; by week 4, reported drop of ~-15.3%.
- Interpretation: “dependency paradox” / overreliance—AI helps now but may undermine users’ independent evaluation ability.

## How is this similar to GALILEO?

- Shared theme: multi-turn interaction dynamics can create *drift* (here: human skill/behavior drift) rather than stable, robust performance.
- Provides an external motivation framing: even when AI improves local outcomes, repeated conversational assistance can produce undesirable longer-horizon effects.

## How is this different from GALILEO?

- Unit of analysis is humans + AI interaction (HCI), not primarily model internal consistency/sycophancy under conversational pressure.
- Outcome is human unaided discernment skill over time, not model susceptibility/robustness metrics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO focuses on controlled, model-centric evaluation of multi-turn susceptibility (e.g., pressure, stance shifts), it can isolate causal factors more cleanly than human-subject longitudinal studies.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a “downstream user impact / overreliance” narrative, this paper is a useful citation to argue why multi-turn robustness matters beyond model scores.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related-work positioning: cite as evidence that conversational AI can induce harmful longitudinal effects (dependency/overreliance), motivating robustness-to-persuasion and stability.
- [ ] Consider adding a short paragraph connecting *model-level* drift/sycophancy to *user-level* drift/overreliance as complementary risks in deployed multi-turn systems.

## Quotes / details to potentially cite

- Abstract (core numbers): AI-assisted sessions improved accuracy by about +21%, while unaided performance on new items declined by week 4 by about -15.3%.
- The paper explicitly frames a “dependency paradox” where assistance provides immediate benefits but undermines longer-term capabilities.
