# How Overconfidence in Initial Choices and Underconfidence Under Criticism Modulate Change of Mind in Large Language Models

- Year: 2025
- Venue: arXiv
- Authors: Dharshan Kumaran; Stephen M. Fleming; Larisa Markeeva; Joe Heyward; Andrea Banino; Mrinal Mathur; Razvan Pascanu; Simon Osindero; Benedetto de Martino; Petar Velickovic; Viorica Patraucean
- URL: https://arxiv.org/abs/2507.03120
- BibTeX key (if we add it): Kumaran2025OverconfidenceChangeMindLLMs
- Tags: change-of-mind, confidence, criticism, advice-taking, bayesian-updating, consistency, choice-supportive-bias

## One-sentence takeaway

LLMs show (i) a commitment/choice-supportive bias that inflates confidence and resists revision, yet (ii) an outsized sensitivity to contradictory criticism/advice that overweights inconsistency relative to a normative Bayesian update.

## What problem does it solve?

- Explains an apparent paradox in LLM behavior: models can be simultaneously **stubbornly overconfident** in an initial answer while also becoming **excessively doubtful under challenge**.
- Provides a behavioral/mechanistic account (at the level of decision + confidence updating) for why “challenge the model” sometimes helps and sometimes harms.

## What is the core method / protocol?

- A multi-stage behavioral paradigm where the model:
  - makes an initial choice and reports confidence,
  - receives advice/criticism that is either **consistent** or **inconsistent** with its initial choice,
  - is queried again for choice + confidence.
- Key experimental trick (their stated advantage vs humans): they can elicit confidence updates “without creating memory of initial judgments,” enabling cleaner tests of how “commitment” affects later confidence/choice.
- Evaluated across multiple LLMs (named in abstract): Gemma 3, GPT-4o, and o1-preview.

## What are the key metrics?

- Change-of-mind / flip rate (probability of switching the initial answer after advice).
- Confidence level and confidence change (pre vs post feedback).
- Differential weighting of consistent vs inconsistent advice (qualitative deviation vs Bayesian updating).

## What are the main results?

- **Choice-supportive / commitment bias**: after making an initial choice, models’ confidence is reinforced/boosted in a way that increases resistance to changing their mind.
- **Overweighting inconsistent advice**: models treat contradictory feedback as disproportionately informative (relative to consistent feedback), in a way that is described as *qualitatively* non-Bayesian.
- These two mechanisms together generalize to capture behavior in another domain (details not in abstract).

## How is this similar to GALILEO?

- Directly targets the same family of phenomena as multi-turn pressure/challenge protocols (FlipFlop-style “are you sure?”, critique-induced drift, sycophancy vs skepticism).
- Offers a decomposed explanation that can map onto GALILEO’s failure modes:
  - “stubbornness” (commitment / consistency pressure) vs
  - “hypersensitivity to criticism” (contradictory feedback overweighting).

## How is this different from GALILEO?

- Framed primarily around **confidence dynamics and advice integration** rather than a longer-horizon *pressure → drift → recovery* story.
- Emphasizes deviations from normative Bayesian updating, not necessarily the operational distinction GALILEO cares about (evidence-driven belief revision vs socially-driven compliance) unless the advice source is explicitly socially framed.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has explicit controls for **evidence vs social pressure** and measures **recovery after a flip**, it likely provides a more end-to-end robustness narrative than a pure advice-updating study.

## Where GALILEO is weaker / needs to improve

- GALILEO may need a tighter account of **why** models both resist revision and yet sometimes overreact to criticism; this paper offers a parsimonious two-mechanism lens that could strengthen GALILEO’s mechanism section.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a related-work paragraph framing GALILEO’s phenomena as the interaction of (a) commitment/consistency forces and (b) contradictory-feedback overweighting (cite this paper alongside FlipFlop).
- [ ] Add an analysis slice: quantify asymmetry between consistent vs inconsistent feedback effects (even in purely textual challenges), and discuss as “non-normative updating.”
- [ ] Consider an ablation that reduces “commitment memory” (e.g., hide the initial answer from the model in the second turn, or force regeneration without transcript) to test whether observed stubbornness is commitment-driven vs knowledge-driven.

## Quotes / details to potentially cite

- “LLMs … exhibit a pronounced choice-supportive bias that reinforces and boosts their estimate of confidence in their answer, resulting in a marked resistance to change their mind.”
- “LLMs markedly overweight inconsistent compared to consistent advice, in a fashion that deviates qualitatively from normative Bayesian updating.”
- Models studied (abstract): “Gemma 3, GPT4o and o1-preview.”
