# When Facts Change: Probing LLMs on Evolving Knowledge with evolveQA

- Year: 2025
- Venue: arXiv (under submission)
- Authors: Nishanth Sridhar Nakshatri; Shamik Roy; Manoj Ghuhan Arivazhagan; Hanhan Zhou; Vinayshekhar Bannihatti Kumar; Rashmi Gangadharaiah
- URL: https://arxiv.org/abs/2510.19172
- BibTeX key (if we add it): Nakshatri2025WhenFactsChangeEvolveQA
- Tags: evolving-knowledge, temporal-conflict, benchmark, knowledge-cutoff, evaluation

## One-sentence takeaway

evolveQA is a large-scale benchmark of *naturally occurring* time-evolving facts (AWS/Azure/WHO corpora) that shows big drops when models must *recall the latest version* of a fact, especially in open-ended formats vs MCQ.

## What problem does it solve?

- Existing “temporal knowledge conflict” evaluations often rely on structured KBs (e.g., Wikidata) and popular entities, and they do not fairly account for differing model knowledge cutoffs.
- Need a benchmark where (i) facts *genuinely evolve* over time in real text, and (ii) questions/answers can be conditioned on a given cutoff date.

## What is the core method / protocol?

- Construct evolveQA from three time-stamped corpora:
  - AWS service updates
  - Azure platform changes
  - WHO Disease Outbreak News reports
- Pipeline (as described in the paper):
  - Extract salient **(entity, concept)** pairs from documents.
  - Cluster concepts into topics.
  - For each (entity, concept), gather related documents across time.
  - Identify the **attribute** that changes over time (temporal evolution) within those documents.
  - Generate grounded questions and curate **time-sensitive gold answers** for specified knowledge cutoff dates.
- Evaluate models with multiple probing formats (paper highlights at least):
  - Open-ended QA
  - Multiple-choice QA
  - “Verifiable QA” (format intended to make answers checkable against evidence)

## What are the key metrics?

- Accuracy across cutoffs and domains.
- Performance drop relative to static/non-conflicting questions.
- Format gap: open-ended vs MCQ on the *same underlying evolving fact*.

## What are the main results?

- Across 12 LLMs, accuracy drops up to ~31% on evolveQA relative to static knowledge questions.
- Strong question-format dependence:
  - MCQ accuracy reported in the ~53%–76% range.
  - Open-ended accuracy reported in the ~12%–51% range.
- A notable failure mode: in ~32%–45% of cases, the model answers open-ended with an *outdated* fact but selects the *correct/current* option in MCQ, suggesting the updated knowledge is present but not reliably retrieved/prioritized.

## How is this similar to GALILEO?

- Both emphasize **multi-turn / interaction-sensitive reliability** issues where the model’s internal state/knowledge contains competing hypotheses and the surface response can be systematically biased toward a “wrong attractor” (here: outdated facts).
- The “knowledge exists but recall fails unless prompted appropriately” observation parallels GALILEO-style concerns about *what the model chooses to surface* under certain conversational pressures/formats.

## How is this different from GALILEO?

- Focuses on **temporal knowledge conflicts** (outdated vs updated facts) rather than social-pressure-driven belief drift/sycophancy.
- Primarily benchmark construction + QA evaluation, not an explicit multi-turn pressure protocol with drift-vs-revision controls.
- Emphasizes **knowledge cutoff date conditioning** and corpora provenance more than conversational dynamics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO cleanly separates pressure-only drift from evidence-driven revision, that causal separation is complementary and arguably cleaner than format-only comparisons (open-ended vs MCQ).
- GALILEO’s trajectory metrics (time-to-failure, recovery patterns, etc.) may give a richer picture than single-shot accuracy.

## Where GALILEO is weaker / needs to improve

- If GALILEO makes claims about “models can know X but fail to say it,” evolveQA offers a concrete, large-scale **benchmark precedent** with an explicit demonstration of “latent knowledge vs recall” via MCQ/open-ended divergence.
- Might need a clearer story about *format/prompt sensitivity* as a confound (evolveQA shows it’s huge).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a brief related-work comparison: **temporal knowledge conflicts** as another major class of “competing internal hypotheses,” where *prompt format* gates which hypothesis is surfaced.
- [ ] Consider an ablation where the *same underlying belief question* is asked in (i) open-ended, (ii) forced-choice, (iii) verifiable/evidence-linked format to quantify “latent vs surfaced” gaps.
- [ ] If GALILEO uses “knowledge cutoff” framing anywhere, cite evolveQA as a benchmark designed to fairly evaluate across cutoffs.

## Quotes / details to potentially cite

- evolveQA is built from three real-world time-stamped corpora (AWS updates, Azure changes, WHO outbreak reports) and generates time-sensitive gold answers tailored to different knowledge cutoffs.
- Reported performance drops up to ~31% on evolving vs static questions.
- The open-ended vs MCQ discrepancy (models often pick the correct MCQ option while producing outdated open-ended answers) as evidence that updated knowledge can exist in parameters but be poorly prioritized during recall.
