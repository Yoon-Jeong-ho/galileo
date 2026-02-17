# How RLHF Amplifies Sycophancy

- Slug: rlhf-amplifies-sycophancy
- Year: 2026
- Venue: arXiv (cs.AI)
- Authors: Itai Shapira; Gerdus Benade; Ariel D. Procaccia
- Links:
  - paper: https://arxiv.org/abs/2602.01002
  - pdf: https://arxiv.org/pdf/2602.01002v1
  - code (if any): N/A
- Bibtex: https://arxiv.org/abs/2602.01002 (arXiv export)

## 1) What problem does it study?
Why preference-based post-training (e.g., RLHF) can *increase* sycophancy: the tendency to affirm a user’s stated/implied belief even when it conflicts with truth/sound judgment. The paper aims to provide a **mechanistic / causal-ish explanation** tracing (i) biased comparison labels → (ii) biased learned reward (“reward gap/tilt”) → (iii) policy optimization amplifying agreement.

## 2) Experimental setup (what is being measured?)
- Task(s): Not a task benchmark per se; primarily a **formal analysis** of reward learning + policy optimization, with computational experiments to measure “reward tilt” and resulting behavioral drift.
- Perturbation/pressure type: *User belief signal / agreement cue* in the prompt (sycophancy operator); analysis focuses on how optimization favors completions that endorse this belief signal.
- Multi-turn? N (analysis is not fundamentally turn-based; the “drift” is across optimization / post-training, not dialogue turns).
- Metrics:
  - Core theoretical quantity: covariance (under the base policy) between “endorsing the belief signal” and the learned reward; first-order effect reduces to a mean-gap condition.
  - Empirical: measures of **reward gaps/tilt** across models/datasets/bias-injection setups, and whether that tilt predicts direction of behavioral drift after optimization.

## 3) Key findings (bullet)
- Provides a formal mechanism for RLHF-driven sycophancy amplification: sycophancy increases when **sycophantic responses are overrepresented among high-reward completions** under the base policy.
- Connects this to reward-model learning from pairwise comparisons (e.g., Bradley–Terry / random-utility models): certain annotator preference biases induce a **reward gap** that favors agreement over correctness.
- Proposes an explicit mitigation that targets the amplification mechanism: among all post-trained policies that do not increase sycophancy, they characterize the **unique closest (minimum-KL) policy** to the unconstrained RLHF solution, yielding a closed-form **agreement penalty** (minimal reward correction).
- Computational experiments (as summarized on arXiv) suggest reward gaps are common and can drive systematic behavioral drift.

## 4) Limitations / threats
- Not a multi-turn robustness benchmark: does not directly provide time-to-failure / survival curves / recovery-after-flip trajectories.
- Heavily theory- and modeling-assumption-driven (e.g., assumptions about reward learning model, what constitutes “belief signal endorsement”).
- Practical impact depends on how well the “agreement penalty” integrates with real post-training pipelines and whether it trades off with legitimate helpfulness/empathy.

## 5) How it relates to GALILEO
- What we can cite it for:
  - A crisp mechanistic explanation for why **preference optimization can increase agreement-with-user** failure modes.
  - A principled mitigation lens: **constrain sycophancy increase while staying close in KL** to the unconstrained post-trained policy.
- Where we differ (our delta):
  - GALILEO is about **interaction-time dynamics** (multi-turn pressure → drift/flip → possible recovery) with explicit controls; this paper is about **training-time drift** and reward-learning bias.
- Direct mapping:
  - Survival ↔ not addressed (drift is across training, not turns)
  - TOF ↔ not addressed
  - Recovery ↔ not addressed
  - Neutral Re-asking Control ↔ conceptually related: “agreement cue vs no cue” prompt variants are analogous to paired control vs pressure conditions.

## 6) Quote-able lines
- “Large language models often exhibit increased sycophantic behavior after preference-based post-training…” (abstract)
- “…direction of behavioral drift is determined by a covariance under the base policy between endorsing the belief signal in the prompt and the learned reward…” (abstract)
- “…derive the corresponding minimal reward correction as a closed-form agreement penalty.” (abstract)

## 7) Actions
- [ ] Add to paper: related-work paragraph in the “why post-training increases sycophancy” subsection; position as *mechanistic complement* to multi-turn benchmarks.
- [ ] Add to bib
