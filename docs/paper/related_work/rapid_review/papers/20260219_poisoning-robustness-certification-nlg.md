# Towards Poisoning Robustness Certification for Natural Language Generation

- Year: 2026
- Venue: arXiv (cs.LG); mentions ICML
- Authors: Mihnea Ghitu; Matthew Wicker
- URL: https://arxiv.org/abs/2602.09757
- BibTeX key (if we add it): Ghitu2026PoisoningCertNLG
- Tags: robustness, certification, poisoning, NLG, shard-and-aggregate, agents, tool-calling, alignment

## One-sentence takeaway

Introduces Targeted Partition Aggregation (TPA), extending shard-and-aggregate poisoning certificates from classification to autoregressive generation by certifying both (i) stability against any change and (ii) validity against targeted harmful tokens/phrases, including multi-turn MILP-tightened guarantees.

## What problem does it solve?

- Existing certified poisoning defenses (e.g., Deep Partition Aggregation / shard-and-aggregate) are designed for classification and become unsound or intractable for autoregressive generation.
- Two key gaps for NLG:
  - **Autoregressive dependency**: certifying token i typically assumes the prefix is fixed; but poisoning can change earlier tokens, invalidating downstream certificates.
  - **Huge output space**: targeted harmful outputs are a tiny subset of an exponentially large sequence space; “no change” (stability) is not the same as “no harmful change” (validity).

## What is the core method / protocol?

- Builds on **shard-and-aggregate**: partition training data into S disjoint shards, train S base models, aggregate votes at inference.
- Formalizes two security properties:
  - **Stability**: robustness to *any* change in generation (untargeted).
  - **Validity**: robustness to *targeted* harmful changes (e.g., a specific token/phrase/class of tool call).
- **Targeted Partition Aggregation (TPA)**:
  - Given ensemble vote counts and a target token/phrase, computes a lower bound on the **minimum poisoning budget** required to make that target win the plurality vote.
  - Intuition: attacker reallocates votes from high-ranked classes to the target in phases; TPA tracks the most efficient vote-reallocation strategy under worst-case shard corruption assumptions.
- Extends beyond first-token:
  - **Sequential multi-token** certification (tight but high latency since ensemble must vote token-by-token).
  - **Phrase-level** certification by treating m-token phrases as labels (trades tightness vs. latency).
  - **Multi-turn / collective certification**: uses an MILP to exploit “budget dilution” across prompts/turns, yielding stronger guarantees when an attacker must succeed repeatedly.

## What are the key metrics?

- Certified radii / budgets: minimum number of poisoned training points needed to (a) change a token (stability) or (b) induce a targeted harmful token/sequence (validity).
- First-token and horizon-style summaries:
  - First Token Stability (FTS@k), First Token Validity (FTV@k)
  - Stability Horizon (SH@k), Validity Horizon (VH@k)
- Practical latency trade-offs as a function of number of shards and inference scheme.

## What are the main results?

- Demonstrates validity certification for **agent tool-calling**:
  - Example described: certifying that tool-call details cannot be forced to a different (harmful/incorrect) tool/parameterization unless the adversary poisons a large fraction of training data.
  - Reports certifying validity under up to **0.5% dataset modification** in one setting (as stated in abstract; details include large sharding).
- Shows certified stability horizons in preference/alignment settings (abstract mentions certifying **8-token stability horizons** in preference-based alignment).
- Notes a major remaining challenge: **inference-time latency** for sequential/large-ensemble certification, with partial fine-tuning explored as a mitigation (at the cost of weaker guarantees).

## How is this similar to GALILEO?

- Both are motivated by **security-sensitive deployment** of LMs/agents and explicitly reason about adversarial robustness guarantees rather than only empirical robustness.
- Connects strongly to **agent tool-calling safety**: certifying that tool calls/parameters cannot be maliciously altered by training-time poisoning.

## How is this different from GALILEO?

- Focuses specifically on **training-time data poisoning certification** via shard-and-aggregate ensembles and vote-margin style bounds.
- Introduces a **targeted** certification objective (validity) tailored to preventing specific harmful generations, rather than only bounding generic output change.
- Heavy reliance on **ensembling/sharding** and formal worst-case assumptions at the shard level; may differ from GALILEO’s core mechanism and threat model focus.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides lower-latency or single-model guarantees (or addresses broader attack surfaces beyond poisoning), it may be operationally simpler than large-shard ensembles with costly certification-time inference.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently lacks a formal notion akin to **validity** (targeted harmful-set robustness), this paper suggests it is an important missing axis: stability-type guarantees can be misaligned with real safety goals in NLG.
- If GALILEO does not cover **multi-turn** certification, the “budget dilution” perspective could tighten guarantees.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider introducing (or mapping to) a **validity-style** guarantee: robustness against membership in a harmful set rather than exact-output stability.
- [ ] For tool-using settings: define harmful sets as “any tool call other than the intended one” or “tool call with unsafe parameter ranges,” then explore whether GALILEO can certify against those.
- [ ] If relevant, cite their formulation as motivation for why *targeted* guarantees matter more than stability in NLG.
- [ ] Consider whether any multi-instance / multi-turn tightening (optimization-based, MILP or otherwise) can strengthen GALILEO’s guarantees.

## Quotes / details to potentially cite

- Abstract framing: classification certificates are "ill-equipped for autoregressive generation" due to sequential prediction and exponentially large output space.
- Defines two properties: **stability** (robustness to any change) and **validity** (robustness to targeted harmful changes).
- Introduces **Targeted Partition Aggregation (TPA)** to compute the minimum poisoning budget required to induce a specific harmful class/token/phrase; extends with **MILP** for tighter multi-turn guarantees.
