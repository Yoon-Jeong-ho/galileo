# Teaching Language Models to Faithfully Express their Uncertainty

- Year: 2025
- Venue: arXiv
- Authors: Bryan Eikema; Evgenia Ilia; José G. C. de Souza; Chrysoula Zerva; Wilker Aziz
- URL: https://arxiv.org/abs/2510.12587
- BibTeX key (if we add it): eikema2025fut
- Tags: uncertainty, calibration, faithfulness, consistency, hedging, qa

## One-sentence takeaway

Faithful Uncertainty Tuning (FUT) fine-tunes an instruction-tuned LLM to *hedge in proportion to its own sample-consistency uncertainty* while (approximately) preserving the underlying answer/semantic distribution.

## What problem does it solve?

- LLMs often present answers with overly decisive language even when the model’s *own* distribution is high-entropy / inconsistent across samples (a “faithfulness gap” between expressed decisiveness and internal uncertainty).
- Existing fixes can require heavy prompt engineering or change the model’s answering behavior; this work targets uncertainty *communication* without changing “beliefs” (the semantic answer distribution).

## What is the core method / protocol?

- Define faithfulness following Yona et al. (2024a): for each response, extract assertions, estimate each assertion’s confidence via **contradiction rate** against other samples (Monte Carlo + NLI judge), and compare to expressed decisiveness inferred from hedges.
- **Synthetic data generation (pushforward construction):** for each QA prompt x:
  - Sample S candidate answers y ~ p(·|x) using *unbiased sampling* (not greedy/beam).
  - Estimate confidence for the sampled answer’s assertion(s) via leave-one-out contradiction rate against the other samples.
  - Map confidence to a human-interpretable hedge level using a verbal–numerical correspondence table (Vogel et al., 2022).
  - Produce a “faithfully hedged” target response via one of:
    - **FUT-interweave:** use an auxiliary LM to rewrite the assertion with the hedge naturally integrated.
    - **FUT-postfix:** append a templated postfix phrase, minimally editing the original answer.
- **Training:** fine-tune the base instruction-tuned model by MLE on these (prompt, faithfully-hedged-response) pairs, aiming to approximate the pushforward distribution that preserves asserted content while adding calibrated hedging.

## What are the key metrics?

- **Faithfulness:** conditional mean faithful generation (**cMFG**, from Yona et al. 2024a), based on per-assertion |decisiveness − confidence|.
- **Task performance:** QA accuracy on open-domain QA.
- **Distribution shift:** total variation distance between *semantic* distributions of answers from base vs tuned model (as a check that FUT doesn’t change what answers are given, only how they’re hedged).

## What are the main results?

- FUT substantially improves cMFG (reduces the faithfulness gap) while largely preserving QA accuracy.
- FUT introduces minimal semantic distribution shift relative to the base model, consistent with the goal “don’t change beliefs, change uncertainty expression”.
- Robustness analyses: improvements hold across decoding strategies, hedger sets, and even alternative uncertainty formats (including numerical expressions).

## How is this similar to GALILEO?

- Shares the theme of **separating latent model state/uncertainty from the surface form** communicated to users, and evaluating robustness/consistency under repeated sampling.
- Uses **multi-sample consistency/contradiction** as a proxy for internal uncertainty, which is often a component in stability / drift / inconsistency analyses.

## How is this different from GALILEO?

- FUT is primarily about *calibrated communication of uncertainty* for (mostly) short-form QA, not about long-horizon conversational pressure, belief drift, recovery, or social influence dynamics.
- Relies on judges (assertion extraction + NLI contradiction detection; plus an LLM-based decisiveness annotator) to build training data and evaluate faithfulness.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets multi-turn dynamics (pressure vs evidence; recovery), it covers failure modes that FUT does not address (FUT focuses on hedging faithfulness for single-turn answers).
- A GALILEO-style protocol can avoid dependence on a particular hedge lexicon / verbal–numerical mapping, instead evaluating behaviorally (e.g., abstain/deferral, stability under challenge).

## Where GALILEO is weaker / needs to improve

- If GALILEO claims “models should express uncertainty faithfully”, FUT is a direct adjacent method + metric stack (cMFG) that reviewers may expect us to cite/compare.
- FUT offers a concrete recipe for *belief-preserving* post-training (pushforward view + TV-distance semantic-shift check) that could strengthen any GALILEO claim about “not changing answers, only presentation”.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider citing FUT + framing “belief preservation” explicitly: measure semantic shift between base and intervention outputs (e.g., TV distance over clustered semantics).
- [ ] If we use repeated sampling/consistency, consider whether a cMFG-like decomposition (confidence vs decisiveness) would clarify results.
- [ ] Add a baseline intervention: postfix-style calibrated hedges (minimal edits) vs more natural rewrites, and check whether this affects downstream user trust / compliance.

## Quotes / details to potentially cite

- Paper framing: repeated queries yield divergent answers, but responses are “typically unhedged or hedged in ways that do not reflect this variability,” creating a “faithfulness gap”.
- FUT data generation: “augmenting model samples with uncertainty hedges … aligned with sample consistency, requiring no supervision beyond the model and a set of prompts.”
- Two strategies: **interweave** (aux LM rewrite) vs **postfix** (template append), trading naturalness vs cost and belief preservation.
