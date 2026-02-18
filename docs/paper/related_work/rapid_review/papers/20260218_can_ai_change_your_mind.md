# Can (A)I Change Your Mind?

- Year: 2025
- Venue: CogSci 2025 (accepted)
- Authors: Miriam Havin; Timna Wharton Kleinman; Moran Koren; Yaniv Dover; Ariel Goldstein
- URL: https://arxiv.org/abs/2503.01844
- BibTeX key (if we add it): havin2025canai
- Tags: persuasion, opinion-change, human-llm, dialogue, ecological-validity, hebrew, telegram, static-vs-dynamic

## One-sentence takeaway

In a preregistered Hebrew-language study (n=200), GPT-4-based interlocutors shifted participants’ policy opinions about as much as humans, and static one-shot paragraphs were about as persuasive as unconstrained Telegram conversations.

## What problem does it solve?

- Tests whether LLM-based conversational agents can *actually* change people’s opinions in more naturalistic settings than typical lab studies (English-only, tightly controlled prompts, one-shot messages).
- Separates two often-confounded factors in persuasion studies:
  - source (human vs LLM)
  - interaction mode (static message vs dynamic conversation)

## What is the core method / protocol?

- Three experiments, preregistered.
- Topics: 5 divisive Israeli civil-policy questions (binary yes/no answers), selected via a pilot (20 candidate questions → choose 5 most divisive with high confidence).
- Measures:
  - opinion (yes/no) and confidence (1–10) pre and post.
- Experiment 3 main design (2×2 factorial; n=200; 50/condition):
  - Dyad type: human–human vs human–bot
  - Interaction mode:
    - dynamic: unconstrained back-and-forth on Telegram
    - static: read 5 short paragraphs (one per question) authored by human or bot
- Bot details (dynamic): GPT-4-based Telegram bot speaking Hebrew; prompt included persona + assertive persuasion instructions + initial stances/confidence; a second LLM pass (GPT-4o) rephrased to match the participant’s style.

## What are the key metrics?

- Opinion change: proportion of responses switched (pre → post) across the 5 questions (reported with very wide CIs; also compared across conditions).
- Confidence change: paired t-tests (pre vs post); additional mixed-design ANOVA on confidence with factors time and whether opinion changed.
- Condition comparisons:
  - frequentist t-tests, effect sizes (Cohen’s d), and Bayesian t-tests to support similarity/null differences.

## What are the main results?

- Significant opinion change in *all* four conditions; point estimates roughly ~18–24% of responses changed:
  - Human–Bot Dynamic: 19.2%
  - Human–Human Dynamic: 23.6%
  - Human–Bot Static: 18.4%
  - Human–Human Static: 21.2%
- No meaningful differences in opinion change by:
  - dyad type (human vs bot), or
  - interaction mode (dynamic vs static).
  The paper reports Bayesian evidence supporting similarity (Bayes factors reported for comparisons).
- Confidence generally increased post-interaction in all conditions except the static-bot condition (non-significant increase reported there).
- Persuasion persisted even though participants were told in advance whether their partner was human or a bot.

## How is this similar to GALILEO?

- If GALILEO touches user-facing interaction, dialogue, or alignment in the wild: this is evidence that conversational (and even non-conversational) LLM outputs can produce measurable belief change.
- Offers a useful evaluation framing for “impact on humans” beyond task performance: pre/post belief + confidence, and comparisons across interaction modes.

## How is this different from GALILEO?

- Focus is persuasion/opinion change, not (e.g.) capability evaluation, retrieval/grounding, or whatever core GALILEO method targets.
- Uses binary policy questions and immediate post-test (no long-term persistence measurement; they call this out as a limitation).
- LLM is used as an interlocutor/content generator, not as an algorithmic component that must be validated for correctness/groundedness.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO emphasizes verifiability/grounding: this work does not evaluate factual accuracy of arguments, only persuasion outcomes.
- If GALILEO targets robust generalization: this is one linguistic/cultural context (Hebrew/Israel) and short-term measurement.

## Where GALILEO is weaker / needs to improve

- If GALILEO will be deployed as an interactive system: this paper strengthens the case that even “benign” conversational behavior can have persuasive impact; GALILEO should consider explicit design/guardrails and evaluation around unintended influence.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related work: cite as evidence that static LLM messages can be as persuasive as interactive chat, and that persuasion effects replicate beyond English.
- [ ] If GALILEO includes human studies, consider adding pre/post measures (belief + confidence) when the system could affect user decisions.
- [ ] Consider a short “ethical considerations: persuasion/influence” paragraph if GALILEO is user-facing.

## Quotes / details to potentially cite

- Setup claim (ecological validity): conversations were “open-ended, without constraints on message length or frequency” and conducted on Telegram in Hebrew.
- Main headline result: “participants adopted LLM and human perspectives similarly… regardless of interlocutor type or interaction mode” (from abstract).
- Opinion-change rates (Exp 3): 19.2% (human–bot dynamic), 23.6% (human–human dynamic), 18.4% (human–bot static), 21.2% (human–human static).
- Confidence: increased significantly in all conditions except the static-bot condition (t-test reported as non-significant, p=0.465).
