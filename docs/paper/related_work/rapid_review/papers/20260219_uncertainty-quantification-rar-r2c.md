# Uncertainty Quantification for Retrieval-Augmented Reasoning

- Year: 2025
- Venue: arXiv
- Authors: Heydar Soudani; Hamed Zamani; Faegheh Hasibi
- URL: https://arxiv.org/abs/2510.11483
- BibTeX key (if we add it): soudani2025r2c
- Tags: rag, rar, uncertainty, uq, abstention, model-selection

## One-sentence takeaway

R2C estimates uncertainty for retrieval-augmented *reasoning* agents by perturbing intermediate reasoning states (thereby changing subsequent retrieval) and measuring answer consistency, yielding better AUROC and improved abstention/model-selection decisions.

## What problem does it solve?

- Existing LLM uncertainty estimation largely ignores uncertainty introduced by *multi-step* retrieval+reasoning loops (RAR), where early retrieval/query-generation mistakes compound.
- Need a black-box-ish UQ signal that reflects both retriever and generator uncertainty across the entire reasoning trajectory.

## What is the core method / protocol?

- Model RAR as an MDP where each intermediate state contains the model “think” + generated search query; actions are retrieve vs answer.
- Compute the most-likely reasoning path and final answer (using low-temperature / greedy-ish decoding).
- Generate multiple *perturbed* rollouts by randomly selecting a reasoning state and applying a perturbation action that alters:
  - query generation,
  - retrieved documents,
  - or the model’s internal “thinking” at a chosen step (paper describes three perturbation actions).
- Because perturbations change the query, retrieval results shift, which changes the next-step generator input; iterating this captures retriever+generator coupled uncertainty.
- Uncertainty score is derived from *consistency* of final answers (majority-vote agreement with the most-likely answer).

## What are the key metrics?

- AUROC for detecting incorrect answers (UQ as a discriminator).
- Extrinsic: abstention metrics (F1Abstain, AccAbstain) when using UQ to decide “I don’t know”.
- Extrinsic: model selection exact match when choosing among multiple RAR systems.
- Efficiency proxies: number of generations needed per query; reported diversity of retrieved docs / queries.

## What are the main results?

- Across five RAR systems and multiple QA datasets, R2C improves AUROC by >5% on average over prior UQ baselines.
- Downstream gains when used as an external signal:
  - Abstention: ~5% improvements in both F1Abstain and AccAbstain.
  - Model selection: ~7% exact match vs single models; ~3% vs selection baselines.
- More “useful” diversity: ~25 unique retrieved docs per uncertainty estimate (vs ~16 for other UQ methods), with higher query diversity; while needing ~3 generations on average (claimed ~2.5x fewer tokens vs baselines using 10 generations).

## How is this similar to GALILEO?

- Aligns with the idea that *trajectory/path* variability is a good proxy for uncertainty in multi-step agentic retrieval/reasoning.
- Treats retrieval and generation as a coupled system and tries to quantify uncertainty due to interaction/feedback across steps.

## How is this different from GALILEO?

- R2C is primarily a *UQ wrapper* around an existing RAR agent (perturb-and-measure-consistency), rather than a new reasoning/retrieval algorithm.
- Uses majority-vote/consistency of final answers as the core uncertainty score; GALILEO may emphasize calibrated scoring, internal signals, or different aggregation (depending on our method choice).
- Perturbations are applied to reasoning states/actions, not necessarily to evidence scoring or provenance constraints.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO already enforces stronger evidence grounding / structured verification, it may reduce the need for multiple perturbed rollouts for reliability.
- If GALILEO produces interpretable uncertainty decompositions (e.g., per-hop evidence confidence), that could be more actionable than a single consistency score.

## Where GALILEO is weaker / needs to improve

- If we do not explicitly model “coupled uncertainty” from retrieval-query-generation feedback, our UQ may miss key failure modes that R2C targets.
- If we rely only on token-probability-based or single-shot UQ, we may underperform in RAR settings.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an experiment: perturb intermediate agent steps (query rewrite / retrieved doc swap / thought perturbation) and measure final-answer agreement as an uncertainty signal.
- [ ] Compare against simple consistency baselines (temperature sampling, prompt perturbation) to isolate the value of *step perturbations that change retrieval*.
- [ ] Consider an ablation: perturb only retrieval vs only reasoning vs both, to quantify where uncertainty originates.
- [ ] If we have abstention/model-selection tasks, report AUROC + F1Abstain/AccAbstain + selection EM for GALILEO-like systems.

## Quotes / details to potentially cite

- “Accurate estimation of UQ for RAR requires accounting for all sources of uncertainty, including those arising from retrieval and generation.”
- Core idea summary: perturb multi-step reasoning states so retriever and generator “continuously reshape one another’s inputs,” then compute uncertainty from answer consistency.
- Reported headline: “improves AUROC by over 5% on average” and downstream gains in abstention/model selection.
