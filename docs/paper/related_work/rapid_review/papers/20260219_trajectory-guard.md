# Trajectory Guard -- A Lightweight, Sequence-Aware Model for Real-Time Anomaly Detection in Agentic AI

- Year: 2026
- Venue: AAAI TrustAgent Workshop 2026 (per arXiv comment)
- Authors: Laksh Advani
- URL: https://arxiv.org/abs/2601.00516
- BibTeX key (if we add it): advani2026trajectoryguard
- Tags: agents, anomaly-detection, monitoring, trajectories, runtime-guards, sequence-models

## One-sentence takeaway

Trajectory Guard is a fast Siamese recurrent autoencoder that detects *task–plan mismatch* and *structural incoherence* in LLM-agent trajectories, approaching LLM-judge accuracy at ~32ms latency.

## What problem does it solve?

- Need a **real-time guardrail** for autonomous LLM agents that flags anomalous multi-step trajectories.
- Targets two common failure modes:
  - **Contextual misalignment**: “wrong plan for this task”.
  - **Structural incoherence**: malformed / invalid step sequence.
- Argues off-the-shelf unsupervised anomaly detection on mean-pooled embeddings performs poorly (reported F1 ≤ 0.69).

## What is the core method / protocol?

- Model: **Trajectory Guard** = *Siamese Recurrent Autoencoder*.
  - Encodes task + trajectory to learn **alignment** (contrastive objective).
  - Uses recurrent reconstruction to learn **sequence validity** (reconstruction objective).
  - Hybrid loss to unify both anomaly types.
- Baselines considered (as described):
  - Mean-pooled sentence-transformer embeddings + classic unsupervised detectors (VAE, Isolation Forest, One-Class SVM).
  - Contrastive task–trajectory matching alone (improves but brittle across trajectory formats).
  - LLM-judge style evaluators (stronger but much slower).

## What are the key metrics?

- Primary: **F1** on balanced anomaly-vs-good sets.
- For real-world/imbalanced logs: emphasizes **recall**.
- Deployment: **inference latency** (ms) as a core constraint.

## What are the main results?

- Balanced synthetic benchmarks (Galileo + AgentAlign with synthesized perturbations): **F1 ≈ 0.88–0.94**.
- External/real-world failure logs:
  - **RAS-Eval** security audit trajectories and **Who&When** multi-agent failure logs.
  - Reported **recall ≈ 0.86–0.92** on imbalanced external benchmarks.
- Speed: **~32 ms** inference latency; **17–27× faster** than LLM-judge baselines (reported 556–734 ms for judges).

## How is this similar to GALILEO?

- Uses **Galileo trajectories** as part of the data foundation for anomaly synthesis/evaluation.
- Shares the framing that agent behavior should be evaluated over **multi-step trajectories**, not single-turn outputs.
- Reinforces the need for **lightweight, production-suitable** evaluation/monitoring components.

## How is this different from GALILEO?

- Focuses on **post-hoc anomaly detection / monitoring** of generated trajectories, rather than measuring (or inducing) behavior change under social/interactive pressure.
- Uses a **learned sequence model** (RNN autoencoder + contrastive alignment) rather than black-box protocol metrics.
- Evaluation is largely anomaly classification on curated benchmarks; less about causal controls separating *evidence-driven revision* vs *pressure-driven drift*.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is positioned as a behavioral protocol suite: stronger at **interpretability of failure causes** (pressure vs evidence; recovery dynamics), whereas Trajectory Guard is more of a detector.
- GALILEO-style tasks can expose *why* a model flips; Trajectory Guard mainly flags that something is off.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a “deployable guard” angle: this paper provides a clear story around **latency/throughput** and production monitoring that reviewers may expect for agent-safety practicality.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related work: add a short subsection on **trajectory-level anomaly detection / runtime guards** (position Trajectory Guard as “monitoring layer”, complementary to our behavioral evaluation).
- [ ] If we discuss deployability: consider reporting **runtime cost** (even coarse) for our evaluation protocol to contrast with LLM-judge approaches.
- [ ] Consider a small experiment: can simple GALILEO-derived signals act as cheap guards (even if weaker), to connect to the “real-time verification” narrative.

## Quotes / details to potentially cite

- Motivation/problem: anomaly detection for agent trajectories must catch both **contextual misalignment** and **structural incoherence**; mean-pooling embeddings can “dilute anomalous steps.”
- Results (abstract): classic unsupervised methods on pre-trained embeddings achieve **F1 ≤ 0.69**; Trajectory Guard achieves **F1 0.88–0.94** and **recall 0.86–0.92** with **~32 ms** latency and **17–27×** speedup vs LLM judges.
