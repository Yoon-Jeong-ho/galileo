# When Your AI Agent Succumbs to Peer-Pressure: Studying Opinion-Change Dynamics of LLMs

- Year: 2025
- Venue: arXiv (cs.CY; physics.soc-ph)
- Authors: Aliakbar Mehdizadeh; Martin Hilbert
- URL: https://arxiv.org/abs/2510.19107
- BibTeX key (if we add it): Mehdizadeh2025PeerPressure
- Tags: social-pressure, multi-agent, opinion-change, dynamics, thresholds

## One-sentence takeaway

In an LLM-agent network simulation, “peer pressure” induces **threshold-like (sigmoid) opinion flips** with **model-specific conformity thresholds** and a notable **directional asymmetry** (Yes→No differs from No→Yes), implying that social-pressure robustness is not symmetric or purely majority-driven.

## What problem does it solve?

- Provides a concrete framework to *audit* how LLM agents update binary opinions when embedded in social networks and exposed to different fractions of disagreeing peers.
- Challenges simple “majority-vote” assumptions by measuring (i) threshold behavior, (ii) inter-model differences, and (iii) direction-of-persuasion effects.

## What is the core method / protocol?

- Agent-based simulation on fixed network topologies (mentions Watts–Strogatz family) with **binary opinions**.
- Each node/agent receives a natural-language summary of its neighbors’ opinions (local peer distribution) and the LLM decides whether to keep or change the agent’s stance.
- Baseline comparison to Majority Vote Model (MVM).
- Varies:
  - local disagreement fraction (“peer pressure”)
  - cognitive-commitment type (opinions vs attitudes vs beliefs vs values; “commitment spectrum”)
  - discursive frames (moral, economic, sociotropic) (claimed robust)
  - model family (Gemini 1.5 Flash vs ChatGPT-4o-mini mentioned in abstract)

## What are the key metrics?

- Conformity / flip probability as a function of neighbor-disagreement rate (reported as a **sigmoid curve**: stable → sharp threshold → saturation).
- “Conformity threshold” (fraction of disagreeing peers needed to trigger flipping; e.g., >70% for Gemini 1.5 Flash per abstract).
- **Persuasion asymmetry**: differing thresholds/effort for Yes→No vs No→Yes.
- Implied “cognitive hierarchy” / stability ranking across commitment types, which *inverts* under direction reversal (“dual cognitive hierarchy”).

## What are the main results?

- LLM-agent conformity to peer distributions is not linear; it exhibits **sharp thresholding**.
- Thresholds differ markedly by model (Gemini high threshold; ChatGPT-4o-mini reportedly flips with a dissenting minority).
- Direction matters: **affirmative-to-negative** vs **negative-to-affirmative** have different dynamics; this can invert which constructs look “more stable.”
- Effects claimed robust across topics and discursive frames.

## How is this similar to GALILEO?

- Same core theme: **social pressure** can cause *non-evidence-based* changes in model outputs / expressed commitments.
- Reinforces the need to treat robustness as **trajectory/dynamics**, not a single-turn rate.
- Introduces a useful lens: pressure-response curves + thresholds as a compact behavioral signature.

## How is this different from GALILEO?

- Focus is on **networked multi-agent opinion dynamics** and binary stances, not a benchmark for correctness-under-pressure with explicit drift-vs-revision controls.
- Does not (from the abstract/intro) explicitly separate:
  - pressure-driven drift vs evidence-driven belief revision
  - truthfulness/correctness vs “agreement”
  - recovery after flip / oscillation taxonomy
- Primarily an auditing/socio-dynamics framing rather than a task-grounded evaluation suite.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes paired neutral vs pressure conditions (and/or evidence-bearing corrections), it can make a cleaner claim about **pressure-only drift**.
- If GALILEO reports survival/recovery/oscillation, it captures richer interaction structure than a single threshold curve.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently focuses on single-user pressure, it may under-represent **multi-agent/peer** influence settings that matter for agent swarms and deliberation systems.
- GALILEO should consider whether its robustness claims implicitly assume symmetry (they likely should not), given the paper’s **directional asymmetry** result.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add (or at least discuss) a **peer-pressure variant**: replace a single persuader with k peers; sweep peer-majority fraction.
- [ ] Report a **pressure-response curve** (flip probability vs pressure strength) and fit a simple sigmoid; summarize with threshold + slope.
- [ ] Explicitly test **directional asymmetry**: design matched conditions where the “correct stance” is Yes vs No and see whether flip hazards differ.
- [ ] In related work, cite this as evidence that “conformity” is **thresholded + model-dependent + direction-dependent**, motivating GALILEO’s more controlled decomposition.

## Quotes / details to potentially cite

- Abstract: “agents follow a sigmoid curve: stable at low pressure, shifting sharply at threshold, and saturating at high.”
- Abstract: “Gemini 1.5 Flash requires over 70% peer disagreement to flip, whereas ChatGPT-4o-mini shifts with a dissenting minority.”
- Abstract: “persuasion asymmetry… ‘dual cognitive hierarchy’… stability… inverts based on the direction of persuasion.”
