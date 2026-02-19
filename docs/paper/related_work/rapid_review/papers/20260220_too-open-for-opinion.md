# Too Open for Opinion? Embracing Open-Endedness in Large Language Models for Social Simulation

- Year: 2025
- Venue: arXiv (position paper)
- Authors: Bolei Ma; Yong Cao; Indira Sen; Anna-Carolina Haensch; Frauke Kreuter; Barbara Plank; Daniel Hershcovich
- URL: https://arxiv.org/abs/2510.13884
- BibTeX key (if we add it): ma2025tooopen
- Tags: social-simulation, open-ended, survey-methodology, evaluation, public-opinion

## One-sentence takeaway

This position paper argues LLM-based social simulation should prefer open-ended (free-text) elicitation and develop evaluation/analysis practices that leverage generative diversity rather than forcing multiple-choice outputs.

## What problem does it solve?

- Many LLM “social simulation” studies use closed-ended or short-answer survey formats because they are easy to score/aggregate.
- The authors argue this design choice (a) collapses nuanced opinions, (b) increases researcher-imposed directive bias/framing, and (c) underuses what LLMs are good at (generating diverse, reasoning-rich responses).

## What is the core method / protocol?

- Conceptual framing + argumentation (not a new model):
  - Defines “open-endedness” for LLM social simulation as prompts that elicit unconstrained free-form responses (vs mapping generations back into fixed labels).
  - Draws parallels to decades of survey methodology on open-ended questions: why they are useful (measurement, exploration, reduced framing, capturing reasoning/heterogeneity), and what challenges they introduce (coding/analysis, variance, evaluation).
  - Calls for practices and evaluation frameworks that treat open-ended generations as a primary signal rather than noise.

## What are the key metrics?

- None proposed/validated empirically in this paper (position paper).
- The paper discusses evaluation needs at a high level (how to analyze/codify open responses; how to design prompts/instruments), but does not introduce a concrete benchmark metric.

## What are the main results?

- Descriptive review claim (from cited reviews): open-text is uncommon in current LLM social simulation work; one statistic reported is that in a follow-up over 53 studies, only 11 (21%) include any open-text component and only 4 (8%) rely primarily on free-form outputs during evaluation.
- Main “result” is the argument/position: open-ended elicitation improves realism and methodological utility by capturing unanticipated views, minority viewpoints, and reasoning processes.

## How is this similar to GALILEO?

- If GALILEO relies on open-ended interactions (free-form responses, explanations, rationale traces), this paper provides writing support for why such design is methodologically important rather than “messy”.
- Motivates evaluation methodology that can handle open-ended outputs (coding, clustering, summarization, qualitative+quantitative hybrids), which often becomes a core burden in open-ended agent/simulation pipelines.

## How is this different from GALILEO?

- No new algorithm/system; it is a position paper synthesizing survey-methodology and NLP perspectives.
- Focus is specifically on *social simulation / public opinion* settings, rather than general agentic evaluation or task performance.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO offers a concrete protocol, dataset, or quantitative evaluation suite for open-ended outputs, it is materially more actionable than the high-level guidance here.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently evaluates via constrained options (multiple-choice, fixed label sets), this paper is a direct critique: it suggests such constraints can suppress heterogeneity and import directive bias.

## Action items for GALILEO (experiments / method / writing)

- [ ] Writing: add a short motivation paragraph (and citation) arguing that open-ended elicitation reduces researcher-imposed directive bias and surfaces unanticipated topics/viewpoints; cite this position paper plus classic survey-method references as needed.
- [ ] Method: if we currently coerce outputs into discrete labels, consider adding an “open-ended first” condition, then post-hoc coding (human or LLM-assisted) to compare how much information is lost.
- [ ] Eval: define/report measures of diversity/heterogeneity and “reasoning richness” for open-ended responses (even if imperfect), alongside traditional aggregate statistics.

## Quotes / details to potentially cite

- The abstract frames the key claim: open-endedness “using free-form text that captures topics, viewpoints, and reasoning processes” is essential for realistic social simulation.
- Reported review statistic: among 53 cataloged studies, only 11 (21%) include any open-text component and only 4 (8%) rely primarily on free-form outputs during evaluation.
