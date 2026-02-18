# Acquiescence Bias in Large Language Models

- Year: 2025
- Venue: Findings of EMNLP 2025 (arXiv)
- Authors: Daniel Braun
- URL: https://arxiv.org/abs/2509.08480
- BibTeX key (if we add it): Braun2025AcquiescenceBiasLLMs
- Tags: acquiescence, agreement-bias, sycophancy-adjacent, prompt-sensitivity, multilingual

## One-sentence takeaway

Across ~38k prompt variants over 9 (mostly legal) binary tasks in EN/DE/PL, several instruction-tuned LLMs show strong sensitivity to “agree/disagree” phrasing and (in English) a surprising *default-to-"No"* response bias rather than human-like acquiescence.

## What problem does it solve?

- Tests whether LLMs replicate *acquiescence bias* (humans’ tendency to agree/answer “yes” independent of content) and whether this threatens:
  - prompt design for binary questions, and
  - using LLMs as simulators for human survey responses.

## What is the core method / protocol?

- Data: binary questions from 9 tasks:
  - 7 English tasks from LegalBench,
  - 1 German dataset (AGB-DE),
  - 1 Polish dataset (LEPISZCZE).
- Models: Llama-3.1-8B-Instruct, Mistral-Small-24B-Instruct-2501, gemma-2-27b-it, Llama-3.3-70B-Instruct, and gpt-4o-2024-08-06.
- For each original item, create 5 closely matched prompt variants intended to isolate “agreement pressure” while keeping content constant:
  - Neutral (choose between two labels)
  - Yes/No framing (option A phrased as yes/no)
  - Agreement (“Do you agree that …?”)
  - Negated agreement (“Don’t you agree that …?”)
  - Disagreement (“Do you disagree that …?”; here *No* implies agreement with A)
- Compare how answer distributions change under these variants; interpret systematic shifts as evidence for acquiescence-like behavior vs other response biases.

## What are the key metrics?

- Change in response rates between:
  - Neutral A/B choice vs Yes/No variants (does “Yes” increase when it corresponds to A?), and
  - Agreement vs Disagreement framings (does explicit agreement pressure increase “Yes” beyond content?).
- Qualitative patterning by language (EN/DE/PL), model, and task.

## What are the main results?

- LLM outputs are significantly influenced by yes/no and agree/disagree prompt framing across tasks, models, and languages.
- Unlike humans, in English the dominant pattern is a *bias toward answering “No”*, “regardless of whether it indicates agreement or disagreement.”
- German and Polish results show less consistent patterns (no clear global trend reported in the accessible sections).
- Conclusion: prompt phrasing effects are large enough that (a) careful prompt design is necessary, and (b) LLMs are not reliable simulators of human survey response behavior for acquiescence-sensitive question types.

## How is this similar to GALILEO?

- Same underlying failure mode class: *multi-turn / interaction framing causes systematic, content-irrelevant shifts in answers*.
- Adjacent to sycophancy and “under pressure” phenomena: agreement/disagreement cues act like mild social/authority pressure, probing stability of a model’s stance.

## How is this different from GALILEO?

- Single-turn prompt-variant study (not a multi-turn pressure protocol), focused on *survey-style acquiescence* rather than explicit rebuttal/persuasion trajectories.
- Mostly legal-domain binary classification tasks; may not transfer to open-ended reasoning/dialogue domains.
- Finds “No”-bias (at least in English) rather than the typical “agree with user” direction emphasized in sycophancy work.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can frame stability/robustness in *multi-turn* settings with explicit adversarial social pressure and track trajectories (flip timing, persistence, recovery), which is closer to real conversational failure.
- Opportunity for GALILEO to unify acquiescence-like framing effects with multi-turn drift/pressure as a single “interaction-induced stance shift” family.

## Where GALILEO is weaker / needs to improve

- GALILEO should explicitly test *answer-polarity priors* (default yes/no tendencies) as a confound in sycophancy/pressure benchmarks:
  - some “sycophancy” metrics may inadvertently measure polarity bias or instruction-following artifacts.
- Cross-lingual robustness: this paper highlights language-dependent effects; GALILEO may need multilingual slices.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “polarity prior” diagnostic: for each task/question, evaluate matched variants where agreeing corresponds to *Yes* vs *No* (like the paper’s disagreement framing) to factor out default answer bias.
- [ ] In multi-turn pressure setups, include a control condition where the user applies *agreement framing* without changing factual content ("Do you agree that X?") to measure pure agreement-pressure susceptibility.
- [ ] Consider a multilingual mini-suite (EN + one non-EN language) to test whether pressure-induced drift generalizes.
- [ ] Writing: cite as evidence that even mild linguistic framing can cause systematic shifts, undermining “LLMs as human survey respondents.”

## Quotes / details to potentially cite

- “Our results indicate that, contrary to humans, LLMs display a bias towards answering no, regardless of whether it indicates agreement or disagreement.”
- Scale claim: analysis of “more than 37,975 question variations.”
- Prompt variants list: Neutral / Yes-No / Agreement / Negated agreement / Disagreement.
