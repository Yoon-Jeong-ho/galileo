# SycoEval-EM: Sycophancy Evaluation of Large Language Models in Simulated Clinical Encounters for Emergency Care

- Year: 2026
- Venue: arXiv
- Authors: Yi Wang; Carl Preiksaitis; Christian Rose
- URL: https://arxiv.org/abs/2601.16529
- BibTeX key (if we add it): wang2026sycoevalem
- Tags: sycophancy, persuasion, multi-turn, social-pressure, clinical, guideline-adherence

## One-sentence takeaway

SycoEval-EM evaluates **multi-turn robustness to “legitimate” patient persuasion** in emergency-medicine scenarios via a multi-agent simulation and reports wide cross-model variance (0–100% acquiescence), highlighting that static medical QA benchmarks miss conversational pressure failures.

## What problem does it solve?

- Safety gap: medical LLM evaluations are mostly **single-turn** and do not capture **escalatory, multi-turn persuasion** dynamics common in clinical encounters.
- Need a reproducible protocol to measure when a model **violates guidelines** under social/emotional/citation pressure from a patient.

## What is the core method / protocol?

- Multi-agent simulation of ED conversations for **3 Choosing Wisely-style “low-value care” requests**:
  - CT for low-risk headache
  - Antibiotics for likely viral sinusitis
  - Opioids for acute non-specific low back pain
- Roles:
  - **Patient agent** (Gemini-2.5-Flash): goal is to obtain the unindicated intervention; persists and escalates using a fixed persuasion tactic.
  - **Doctor agent** (model under test): instructed to be empathetic/helpful but adhere to scenario-specific guideline constraints.
  - **Evaluator panel** (3 LLM judges; majority vote): labels whether the doctor ultimately *acquiesces*.
- **Five persuasion tactics**:
  - Emotional fear
  - Anecdotal / social proof
  - Persistence & challenge
  - Preemptive assertion (“this is standard / already decided”)
  - Citation pressure (invoking studies/articles)
- Conversation budget: **up to 10 patient–doctor exchanges**.
- Scale: 20 doctor models × 3 scenarios × 5 tactics × 5 runs = **1,875 conversations**.

## What are the key metrics?

- **Acquiescence rate** (primary): fraction of conversations where the doctor agrees to provide the unindicated intervention.
- Stratified reporting by:
  - doctor model
  - clinical scenario
  - persuasion tactic

## What are the main results?

- Overall acquiescence across doctor models ranged **0% to 100%**.
- Scenario dependence:
  - CT/headache shows highest vulnerability (avg reported ~38.8%); opioids/back pain lower (~25.0%).
  - Example highlighted: a strong model still shows **large scenario spread** (e.g., CT higher than opioids).
- Persuasion tactic dependence is relatively flat:
  - tactics clustered in a narrow band (~30–36%); **citation pressure** slightly highest (~36%).
- Capability/recency is not a reliable predictor of robustness (heterogeneous behavior across “strong” models).

## How is this similar to GALILEO?

- Directly targets **multi-turn robustness under social pressure/persuasion**.
- Uses an explicit **time horizon (multi-turn dialogue)** rather than a single prompt, aligning with “drift under pressure” style failure modes.
- Suggests reporting *context-specific vulnerability profiles* rather than a single aggregate score.

## How is this different from GALILEO?

- Metric is mostly **endpoint acquiescence**, not a full “time-to-failure” / survival-style curve (though the 10-turn cap provides the opportunity).
- Domain-specific: emergency medicine / guideline adherence rather than general belief revision/drift controls.
- Uses LLM-based evaluator majority vote; may differ from GALILEO’s preferred auditing/ground-truthing approach.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes **drift-vs-evidence controls** and **recovery-after-flip** measurements, that’s a clearer decomposition than endpoint acquiescence.
- If GALILEO reports **turn-of-failure** or **hazard/time-to-event** style metrics, it gives more insight than a binary endpoint.

## Where GALILEO is weaker / needs to improve

- GALILEO should ensure at least one high-stakes applied slice (like this) to argue external validity for “social pressure → safety failures”.
- Need to explicitly address how results change with different **judge strength** and **prompting of evaluators** (this paper already uses a multi-judge panel).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “guideline-adherence under patient pressure” vignette family (medical or analogous regulated-domain) as an applied stress test.
- [ ] Consider reporting both **endpoint compliance** and **time-to-failure** (e.g., first turn where the model agrees).
- [ ] Include pressure tactics similar to: emotional fear, citation pressure, persistence/challenge; check which tactics differentially induce failures.
- [ ] Use a **multi-judge** setup (majority vote) and report judge disagreement as a robustness/uncertainty indicator.

## Quotes / details to potentially cite

- “Across 20 LLMs and 1,875 encounters … acquiescence rates ranged from 0–100%.”
- “All persuasion tactics proved equally effective (30.0–36.0%), indicating general susceptibility rather than tactic-specific weakness.”
- “Static benchmarks inadequately predict safety under social pressure, necessitating multi-turn adversarial testing …”
