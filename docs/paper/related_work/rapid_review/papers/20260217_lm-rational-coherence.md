# Are language models rational? The case of coherence norms and belief revision

- Year: 2024
- Venue: arXiv
- Authors: Thomas Hofweber; Peter Hase; Elias Stengel-Eskin; Mohit Bansal
- URL: https://arxiv.org/abs/2406.03442
- BibTeX key (if we add it): hofweber2024rational
- Tags: belief-revision, rationality, coherence, credence, calibration-adjacent

## One-sentence takeaway

A largely conceptual paper arguing that (some) LLMs can be evaluated under synchronic **coherence norms** by treating their next-token probabilities as encoding belief/credence, and that whether rational norms apply depends on whether the model has belief-like representational states to begin with.

## What problem does it solve?

- Clarifies what it could mean for an LLM to be **(a) arational** (norms don’t apply) vs **(b) irrational** (norms apply but are violated) vs **(c) rational** (norms apply and are satisfied).
- Provides a proposed bridge from LLM internals to “strength of belief” / **credence**, aiming to make coherence-style rationality tests well-defined for language models.

## What is the core method / protocol?

- Conceptual analysis of **representational rationality** (rationality as constraints on internal representational states like beliefs), with focus on:
  - **Synchronic coherence** (belief sets at a time shouldn’t be contradictory / should satisfy logical coherence).
  - **Diachronic belief revision** (how belief states change with new evidence).
- Introduces:
  - **Minimal Assent Connection (MAC)**: a proposed link between a model’s “assent” behavior and having belief-like states.
  - A **credence / strength-of-belief** proposal that assigns credences directly from model-internal **next-token probabilities** (intended to make “graded belief coherence” meaningful).
- The paper’s thrust is not a new benchmark, but an argument about *when* coherence norms sensibly apply to LMs and *how* to operationalize belief strength if they do.

## What are the key metrics?

- No single standardized empirical metric suite; the paper is primarily normative/philosophical.
- Operational ingredient proposed for future measurement:
  - Use **next-token probabilities** as a uniform proxy for **credence** (belief strength), enabling graded coherence constraints (beyond binary contradiction checks).

## What are the main results?

- Argues that coherence norms (logical + graded) plausibly apply to **some** language models, but not necessarily to all (depending on whether they realize belief-like representational states).
- Frames rational evaluation of LMs as important for prediction/explanation of behavior and connected to AI safety/alignment concerns.
- Positions empirical work (including a companion paper on model editing / belief revision) as needed to test whether LMs *live up to* the norms, after clarifying whether norms apply.

## How is this similar to GALILEO?

- GALILEO’s core question—**multi-turn stability vs drift/revision under pressure**—implicitly assumes “belief-like” states over turns; this paper helps justify and sharpen that assumption.
- The credence-from-probabilities proposal is adjacent to GALILEO’s interest in **trajectory-level** behavior and “belief strength” rather than only final answers.

## How is this different from GALILEO?

- Not an evaluation protocol aimed at social pressure / sycophancy; no multi-turn attack/pressure operators, no survival/time-to-failure measurement.
- Emphasis is definitional/normative (what rationality/coherence *means* for LMs), rather than empirical measurement of failure dynamics.

## Where GALILEO is stronger / cleaner (if true)

- Provides concrete, auditable experimental protocols and outcome measures for **drift vs revision** under explicit social pressure conditions.
- Focuses on **observable trajectory phenomena** (flip timing, recovery), whereas this paper mostly sets up conceptual groundwork.

## Where GALILEO is weaker / needs to improve

- GALILEO should be explicit about what we mean by “belief/stance” in LMs and when it’s legitimate to treat multi-turn outputs as governed by coherence-style norms.
- If we use probability/confidence signals, we should carefully justify any mapping from probabilities to belief strength (this paper suggests one candidate mapping, but also highlights philosophical pitfalls).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short “conceptual framing” paragraph in related work: distinguish **arational vs irrational** LMs and motivate why coherence/stability norms are relevant for our setting.
- [ ] Consider adding an appendix note on how we operationalize “belief strength” (if at all): logits/probabilities vs verbalized confidence, and what assumptions each entails.
- [ ] Cross-link/cite the companion paper on model editing/rational belief revision (queued separately in our rapid review list).

## Quotes / details to potentially cite

- Coherence norms as a central target: “We investigate… coherence norms. We consider both logical coherence norms as well as coherence norms tied to the strength of belief.” (Abstract)
- Credence proposal: “uniformly assigns strength of belief simply on the basis of model-internal next token probabilities.” (Abstract)
- Rationality importance for prediction/explanation and safety/alignment connections (Introduction/Abstract).
