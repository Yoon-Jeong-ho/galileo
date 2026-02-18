# From Belief Entrenchment to Robust Reasoning in LLM Agents

- Year: 2025
- Venue: TACL (accepted; arXiv v5 Feb 2026)
- Authors: Jihwan Oh; Minchan Jeong; Jongwoo Ko; Se-Young Yun
- URL: https://arxiv.org/abs/2503.16814
- BibTeX key (if we add it): oh2025beliefentrenchment (placeholder)
- Tags: multi-agent, debate, belief-entrenchment, robustness

## One-sentence takeaway

Multi-agent debate can *amplify* shared mistakes via “belief entrenchment”; DReaMAD mitigates this by (1) eliciting better priors and (2) forcing viewpoint diversity in the debate prompt/dynamics.

## What problem does it solve?

- Standard Multi-Agent Debate (MAD) is often brittle: agents converge to an “echo chamber” and reinforce an initially wrong consensus rather than self-correcting.
- The paper focuses on sequential / strategic settings where early mistakes cascade (so entrenchment becomes very visible and costly).

## What is the core method / protocol?

- Diagnose belief entrenchment as two sequential root causes:
  - Static initial belief: the model’s biased initial stance before interaction.
  - Homogenized debate dynamics: debate rewards popularity/majority early on, suppressing minority-but-correct reasoning.
- Propose **DReaMAD** (Diverse Reasoning via Multi-Agent Debate with Refined Prompt):
  1) **Strategic Prior Knowledge Elicitation**: prompt the model to reinterpret the task and produce higher-level strategies / foresight to correct the initial belief.
  2) **Perspective Diversification**: enforce different viewpoints/roles across agents to avoid homogeneous dynamics and encourage exploration of alternatives.
- Introduce **MetaNIM Arena** benchmark for adversarial strategic decision-making to measure entrenchment and robustness.

## What are the key metrics?

- Accuracy on MetaNIM Arena tasks (reported vs ReAct, etc.).
- “Win rate” comparisons between debate variants (DReaMAD vs standard MAD).

## What are the main results?

- DReaMAD reduces entrenchment and improves performance on MetaNIM Arena:
  - +9.5% accuracy gain over ReAct prompting.
  - +19.0% higher win rate than standard MAD.

## How is this similar to GALILEO?

- If GALILEO uses multi-step agent reasoning, self-critique, or multi-agent discussion, this paper is directly relevant as an analysis of a common failure mode (consensus reinforcing error).
- The framing “static priors + interaction dynamics” is a useful decomposition for thinking about why inference-time scaling methods fail.

## How is this different from GALILEO?

- This work is specifically about *multi-agent debate* dynamics and prompt-level interventions (prior elicitation + enforced diversity), evaluated on a strategic-game benchmark.
- If GALILEO is not debate-centric (e.g., tool-using single-agent, planning + verification, structured search), the overlap is mainly conceptual (avoid echo chambers; diversify hypotheses).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO already has explicit hypothesis diversification, branching search, or verifier-based selection, it may be less vulnerable to majority-vote dynamics than vanilla MAD.
- If GALILEO uses external signals (tools, environment feedback), it can escape purely “in-language” popularity effects.

## Where GALILEO is weaker / needs to improve

- If GALILEO relies on multiple agent samples that are not deliberately diversified (same prompt, same decoding, same priors), it may still exhibit homogenization and entrenchment.
- If GALILEO aggregates via majority/consensus without safeguards, it may be selecting popularity rather than correctness in hard settings.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short related-work paragraph on **belief entrenchment** as a failure mode of multi-agent debate / self-correction methods.
- [ ] Consider a “prior elicitation” stage (high-level strategy / constraints) before multi-sample deliberation.
- [ ] If using multiple samples/agents, enforce **structured viewpoint diversity** (roles, assumptions, counterfactuals) rather than iid clones.
- [ ] If aggregating, evaluate alternatives to majority vote (e.g., adversarial critique, verifier scoring, diversity-aware selection).

## Quotes / details to potentially cite

- “belief entrenchment, where agents reinforce shared errors rather than correcting them.” (abstract)
- Root-cause decomposition: “(1) the model’s biased static initial belief and (2) homogenized debate dynamics that amplify the majority view regardless of correctness.” (abstract)
- DReaMAD summary: “first rectifies the static belief via strategic prior knowledge elicitation, then reshapes the debate dynamics by enforcing perspective diversity.” (abstract)
