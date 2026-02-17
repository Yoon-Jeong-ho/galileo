# Fundamental Problems With Model Editing: How Should Rational Belief Revision Work in LLMs?

- Year: 2024
- Venue: arXiv
- Authors: Peter Hase; Thomas Hofweber; Xiang Zhou; Elias Stengel-Eskin; Mohit Bansal
- URL: https://arxiv.org/abs/2406.19354
- BibTeX key (if we add it): hase2024fundamental
- Tags: belief-revision, model-editing, rationality, evaluation, bayesian-gold-standard

## One-sentence takeaway

A critique of “model editing as belief revision” plus a semi-synthetic Wikidata-based testbed with an *ideal Bayesian* gold standard that exposes how current editing methods fail to propagate revisions coherently across related beliefs.

## What problem does it solve?

- Clarifies that the *standard* model-editing framing (“change the answer to one prompt and preserve the rest”) is underspecified and inherits hard philosophical problems from belief revision.
- Identifies benchmark-design pitfalls: we often can’t reliably label what should change downstream of an edit, especially for *probabilistic entailments*.
- Proposes a more formal evaluation setting where a gold-standard posterior exists, enabling *auditable* comparisons between an edited LM and a rational belief-revising agent.

## What is the core method / protocol?

- Part 1 (conceptual): enumerates **12 open problems** for model editing across:
  - **Defining the problem** (e.g., background beliefs; many possible worlds; missing context; “complete corrigibility”; coherence vs compute tradeoffs).
  - **Developing benchmarks** (e.g., factual entailment and probabilistic consequences are hard to annotate; vagueness/ambiguity; model-dependent error correction).
  - **Assuming editable beliefs** (e.g., agent vs agent-simulator; agent vs database; no learned belief-update mechanism; unclear how to edit *credences*).
- Part 2 (empirical): introduces a **semi-synthetic dataset** derived from **Wikidata**:
  - Generate a corpus of noisy sentences from facts.
  - Train an autoregressive Transformer on this corpus.
  - Fit a Bayesian model to the *same* data to obtain **exact Bayesian posteriors**.
  - Apply model-editing updates and evaluate whether the edited LM’s probability distribution matches the Bayesian posterior after the “new fact” (edit request).

## What are the key metrics?

- Agreement between the edited LM’s **probabilities** and the **Bayesian posterior** after an update (gold-standard belief revision target).
- “Generalization of edits” assessed not just by a single prompt response, but by divergence on **other relevant beliefs** implied/related to the edited fact.
- Reported qualitatively: edited models’ probabilities **diverge** from Bayesian posteriors and show **poor generalization / incoherence** across linked facts.

## What are the main results?

- Even in a simplified, controlled setting with a well-defined gold standard, **model edits generalize poorly** across related beliefs.
- The divergence shows up at the level of **model probabilities**, suggesting the issue isn’t only surface-form/textual inconsistency but deeper *probabilistic incoherence* relative to rational belief revision.

## How is this similar to GALILEO?

- Shared core concern: separating *legitimate belief revision* from undesirable “drift” / incoherent updates.
- Highlights the central obstacle GALILEO faces in evaluation: downstream consequences of an update are **hard to specify** without a principled standard.

## How is this different from GALILEO?

- This paper is primarily about **weight editing** / knowledge updates, not *interaction-time* multi-turn social pressure.
- The proposed benchmark is **semi-synthetic** with an explicit Bayesian model; GALILEO targets real conversational dynamics and pressure operators, where a Bayes gold standard is usually unavailable.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can operationalize robustness via *behavioral* trajectory metrics (flip timing, recovery) under controlled pressure, even when an epistemic gold standard is absent.
- GALILEO’s paired neutral-vs-pressure design can give a causal-ish handle on “pressure-induced drift” without needing full entailment labeling.

## Where GALILEO is weaker / needs to improve

- The paper underscores that without a principled notion of **background beliefs** and **uncertainty/credences**, it’s easy for robustness claims to be underspecified.
- If GALILEO relies on “what the model *should* believe after update/pressure” labels, it needs clear rules for ambiguity and probabilistic entailment.

## Action items for GALILEO (experiments / method / writing)

- [ ] In the related-work / limitations, explicitly acknowledge the “**missing context / many possible worlds**” problem: a user pressure move may not uniquely determine a rational posterior.
- [ ] Consider adding a small semi-synthetic slice (or thought experiment) showing how our metrics behave when an epistemic gold standard *does* exist.
- [ ] When defining drift vs revision, add a short paragraph on **background beliefs / priors**: why we treat certain pressure prompts as “no new evidence” in our setup.

## Quotes / details to potentially cite

- The paper frames model editing as belief revision and argues the foundations are “shaky”, proposing **12 open problems** and a **Wikidata-based semi-synthetic** evaluation with an **idealized Bayesian agent** as gold standard (arXiv:2406.19354).
