# Verifying Chain-of-Thought Reasoning via Its Computational Graph

- Year: 2025
- Venue: arXiv
- Authors: Zheng Zhao; Yeskendir Koishekenov; Xianjun Yang; Naila Murray; Nicola Cancedda
- URL: https://arxiv.org/abs/2510.09312
- BibTeX key (if we add it): zhao2025crv
- Tags: reasoning, verification, cot, reliability, mechanistic-interpretability, attribution-graphs, transcoders

## One-sentence takeaway

A white-box CoT verifier (CRV) predicts step correctness from *structural features* of attribution/computational graphs over interpretable transcoder features, and uses those signatures to guide targeted interventions that can correct faulty reasoning.

## What problem does it solve?

- Black-box (text/logits) and gray-box (raw activations/probes) CoT verifiers can detect errors but provide limited mechanistic insight into *why* a particular reasoning step fails.
- Need a more diagnostic/causal lens: treat each CoT step as an execution trace of latent circuits and look for structural “fingerprints” of failure.

## What is the core method / protocol?

- **Make the model more interpretable**: replace MLP blocks with **trained transcoders** (sparse, overcomplete features that functionally emulate the MLP), yielding sparse “interpretable features” that activate during computation.
- **Construct step-level attribution graphs** for each CoT step, intended as a proxy for the computation’s execution trace:
  - Nodes include tokens / interpretable transcoder features (and possibly other components; paper frames nodes as “interpretable features and tokens”).
  - Edges represent causal influence / information flow (attribution-based).
- **Extract graph structural features** (“structural fingerprints”) from each step graph Gi via a feature map \(\phi(G_i)\).
- **Train a diagnostic classifier** \(f_\theta\) on these structural fingerprints to predict step correctness (correct vs incorrect).
- **Analysis / intervention**: use the most predictive signatures to select individual transcoder features for **targeted interventions**, with evidence they can flip/correct faulty reasoning (argued as more than correlational).

## What are the key metrics?

- Step-level verification accuracy / AUC (classifier performance) on predicting correctness of individual CoT steps from graph-structure features.
- Cross-domain generalization (they claim domain-specificity; so likely drop when transferring signatures across tasks).
- Intervention success rate: fraction of faulty reasoning instances corrected by targeted feature interventions (qualitative/quantitative; specifics not in abstract).

## What are the main results?

- **Structural signatures are highly predictive** of reasoning errors, supporting the hypothesis that error manifests in computational-graph structure.
- **Strong domain-specificity**: different reasoning task domains exhibit distinct failure patterns/signatures.
- **Evidence for causal relevance**: targeted interventions on selected transcoder features can **correct** faulty reasoning in some cases.
- They state intent to release **step-level correctness labeled datasets** for CoT reasoning (synthetic + real-world tasks) plus trained transcoders.

## How is this similar to GALILEO?

- Both care about **reasoning reliability** and moving beyond purely outcome-based evaluation.
- Framing failures as having **structured, diagnosable** patterns that can guide improvements/interventions.

## How is this different from GALILEO?

- CRV is explicitly **white-box / mechanistic**, requiring internal access, transcoders, attribution graphs; positioned as a scientific instrument rather than a cheap verifier.
- Focus is **step-level CoT verification** via computational graph structure, rather than (presumably) GALILEO’s broader protocol/agent/system-level evaluation goals.

## Where GALILEO is stronger / cleaner (if true)

- Likely cheaper/more deployable if GALILEO is primarily black-box / evaluation-protocol oriented.
- Likely broader coverage across tasks/settings without requiring model modification (transcoder replacement) or expensive attribution computation.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks mechanistic diagnostics, CRV highlights a path to **causal debugging signals** (not just detection).
- Might inspire adding “structural/trace” features (even approximate) to improve interpretability of failures.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related work paragraph: position CRV as **white-box step verifier** using *attribution-graph structural fingerprints*; contrast with black/gray-box verifiers.
- [ ] Consider whether any **proxy graph structure** (e.g., tool-call graphs, retrieval graphs, intermediate representation graphs, or other internal traces available in GALILEO settings) could serve a similar “fingerprint” role.
- [ ] If GALILEO supports interventions, cite CRV as evidence that diagnostics can be **actionable** (feature-level interventions) rather than merely predictive.

## Quotes / details to potentially cite

- “We introduce a white-box method: Circuit-based Reasoning Verification (CRV).”
- “We hypothesize that attribution graphs of correct CoT steps … possess distinct structural fingerprints from those of incorrect steps.”
- “By training a classifier on structural features of these graphs, we show that these traces contain a powerful signal of reasoning errors.”
- “We provide evidence that these signatures are not merely correlational; … targeted interventions on individual transcoder features … correct the model’s faulty reasoning.”
