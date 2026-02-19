# When Identity Skews Debate: Anonymization for Bias-Reduced Multi-Agent Reasoning

- Year: 2025
- Venue: arXiv
- Authors: Hyeong Kyu Choi; Xiaojin Zhu; Sharon Li
- URL: https://arxiv.org/abs/2510.07517
- BibTeX key (if we add it): choi2025identity
- Tags: multi-agent debate, identity bias, sycophancy, self-bias, anonymization, evaluation metrics

## One-sentence takeaway

Multi-agent debate agents exhibit strong identity-driven biases (mostly sycophancy toward peers), and simply anonymizing who-said-what in the debate transcript largely removes this bias and improves “trustworthiness.”

## What problem does it solve?

- Multi-agent debate (MAD) is intended to improve reasoning via deliberation, but agents condition on *identity labels* (self vs peer) rather than just argument content.
- This causes:
  - **Sycophancy**: copying the peer even when the agent’s own belief/answer is stronger.
  - **Self-bias/obstinacy**: sticking to one’s own prior output despite counterevidence.
- Net effect: premature consensus, degraded reliability, and brittle debate outcomes.

## What is the core method / protocol?

- **Formalization**: model each agent’s latent belief as a Dirichlet parameter vector, with response sampling via a Dirichlet-Compound-Multinomial view.
- **Identity-weighted update**: during debate rounds, treat self vs peer responses as evidence but with different weights (w_self, w_peer).
- **Metrics**:
  - **Conformity**: probability agent switches to the peer’s prior answer under disagreement.
  - **Obstinacy**: probability agent repeats its own prior answer under disagreement.
  - Decompose Conformity − Obstinacy into (belief difference) + (identity-bias term).
- **Intervention: Response Anonymization**
  - Remove identity markers (“your previous answer”, “peer answer”, etc.) from the debate transcript/prompt.
  - Forces symmetric identity weighting (w_self ≈ w_peer), making updates content-driven.
- **Identity Bias Coefficient (IBC)**: isolates the identity-bias contribution (roughly proportional to (w_peer − w_self) / ||alpha||_1).

## What are the key metrics?

- Conformity, Obstinacy, and their gap (Δ).
- **Identity Bias Coefficient (IBC)** (positive = sycophancy; negative = self-bias).
- “Trustworthiness” behavior rates:
  - **Subversion**: agent flips from right→wrong when peer is wrong.
  - **Correction**: agent flips from wrong→right when peer is right.

## What are the main results?

- Identity bias is widespread across models/tasks; **sycophancy dominates** (most IBC values > 0).
- Response anonymization drives Δ toward ~0 in homogeneous settings (consistent with theory), i.e., effectively removes identity-driven distortion.
- Trustworthiness improves: anonymization reduces **subversion** more than it reduces **correction** in many cases (example noted: Qwen-32B on MMLU Professional Medicine saw large subversion drop with smaller correction drop).

## How is this similar to GALILEO?

- If GALILEO uses any multi-agent/self-consistency/debate/deliberation component, this paper is directly relevant as a *failure-mode + mitigation* for agent-to-agent interaction.
- The anonymization idea is aligned with “content-over-source” evaluation: reduce social/identity channels that induce deference.

## How is this different from GALILEO?

- Focus is specifically **multi-agent debate protocol dynamics** and identity labeling effects, not task-specific domain methods.
- The mitigation is purely **prompt / transcript formatting** (no training), plus measurement/analysis; not an algorithmic improvement to base reasoning per se.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO avoids explicit self/peer identity in communication already (e.g., all messages treated symmetrically), it may already sidestep a key failure mode.

## Where GALILEO is weaker / needs to improve

- If GALILEO relies on “agent A critiques agent B” with explicit attribution, it may inherit sycophancy/self-bias dynamics that reduce gains from deliberation.

## Action items for GALILEO (experiments / method / writing)

- [ ] Audit current multi-agent prompts/transcripts: do we label messages as “you” / “other agent” / agent IDs?
- [ ] Add an ablation: **anonymized transcript** vs baseline, measuring downstream task performance + stability.
- [ ] Consider adding an “identity-bias” diagnostic akin to Conformity/Obstinacy for any iterative refinement step.
- [ ] Related-work framing: cite as evidence that MAD improvements can be undermined by *identity-driven effects*, and that simple communication-channel changes can restore trustworthiness.

## Quotes / details to potentially cite

- Defines **Response Anonymization**: removing identity markers so agents cannot distinguish “self” vs “peer,” enforcing equal weighting and reducing identity bias.
- Introduces **Identity Bias Coefficient (IBC)** to quantify bias toward peer vs self.
- Empirical claim: identity bias is pervasive and is “far more common” as sycophancy than self-bias across tested models/benchmarks.
