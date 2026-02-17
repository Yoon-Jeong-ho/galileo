# Disentangling Ambiguity from Instability in Large Language Models: A Clinical Text-to-SQL Case Study

- Year: 2026
- Venue: arXiv
- Authors: Angelo Ziletti (arXiv listing; full author list not captured from abstract page)
- URL: https://arxiv.org/abs/2602.12015
- BibTeX key (if we add it): ziletti2026clues
- Tags: uncertainty, ambiguity, instability, clarification, triage, clinical, text-to-sql

## One-sentence takeaway

CLUES decomposes output uncertainty into (user) ambiguity vs (model) instability for Text-to-SQL, enabling targeted interventions like clarification vs human review/model improvement.

## What problem does it solve?

- In clinical Text-to-SQL deployments, multiple distinct SQL outputs can occur for a prompt.
- Not all diversity is the same: some is because the *input is underspecified* (ambiguity, should ask clarifying questions), and some is because the *model is unreliable* (instability, should route to human review / improve model).
- Prior single-score uncertainty measures do not separate these regimes.

## What is the core method / protocol?

- Frame Text-to-SQL as a 2-stage process: **interpretations → answers**.
- Build a bipartite semantic graph relating sampled/possible interpretations to produced answers.
- Define two uncertainty components:
  - **Ambiguity score**: uncertainty attributable to multiple plausible interpretations.
  - **Instability score**: uncertainty attributable to model variability conditional on interpretation.
- Compute the instability score using a **Schur complement** on a matrix representation of the bipartite graph.
- Use the resulting 2D “uncertainty regimes” (low/high ambiguity × low/high instability) to prescribe actions:
  - high ambiguity → query refinement / clarification
  - high instability → human review / model improvement

## What are the key metrics?

- Failure prediction / error triage performance (paper claims improvements vs prior uncertainty scoring).
- Coverage vs error capture in deployment-like triage (e.g., fraction of errors captured within top-risk bucket).

## What are the main results?

- On AmbigQA / SituatedQA (with gold interpretations) and a clinical Text-to-SQL benchmark (with known interpretations), CLUES improves failure prediction over Kernel Language Entropy (KLE).
- In a deployment setting (where interpretations are not gold), CLUES remains competitive while additionally providing the ambiguity/instability breakdown.
- The **high-ambiguity/high-instability** regime reportedly contains **51% of errors** while covering **25% of queries**, suggesting efficient triage.

## How is this similar to GALILEO?

- Similar spirit: turn “LLM uncertainty / reliability” into actionable signals for evaluation and routing (e.g., deciding when to intervene).
- Emphasizes *diagnostic decomposition* rather than a single scalar confidence.

## How is this different from GALILEO?

- Very task- and representation-specific (Text-to-SQL; interpretations→answers graph), whereas GALILEO may aim for a more general evaluation framework.
- Uses a graph/matrix (Schur complement) construction; not just prompting-based self-consistency or entropy-style uncertainty.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides task-agnostic protocols and standardized reporting, it may be easier to reuse across domains than CLUES’ two-stage interpretation modeling.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently uses a single uncertainty score, this paper suggests value in explicitly separating **ambiguity** (needs clarification) from **instability** (needs oversight/improvement).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “ambiguity vs instability” framing in related work for uncertainty/triage.
- [ ] Consider reporting a 2-axis decomposition in GALILEO experiments (even if approximated), mapping regimes to interventions.
- [ ] If GALILEO has a triage component, compare against CLUES-style error capture at fixed query coverage.

## Quotes / details to potentially cite

- “Deploying large language models for clinical Text-to-SQL requires distinguishing … (i) input ambiguity … and (ii) model instability …” (abstract)
- “CLUES … decomposes semantic uncertainty into an ambiguity score and an instability score.” (abstract)
- “The high-ambiguity/high-instability regime contains 51% of errors while covering 25% of queries …” (abstract)
