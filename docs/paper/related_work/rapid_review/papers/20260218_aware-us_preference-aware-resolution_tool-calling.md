# AWARE-US: Benchmark for Preference-Aware Resolution in Tool-Calling Agents

- Year: 2026
- Venue: arXiv
- Authors: Mehmet Kurmaz
- URL: https://arxiv.org/abs/2601.02643
- BibTeX key (if we add it): kurmaz2026awareus
- Tags: tool-calling, multi-turn, preference-inference, infeasibility, query-relaxation, benchmark

## One-sentence takeaway

AWARE-US frames empty-result handling for tool-calling agents as *preference-aware query repair* (relax the least-important constraint) and provides a persona-grounded benchmark + simple weighting/ranking baselines.

## What problem does it solve?

- Tool-calling / task-oriented dialogue agents often:
  - **lack constraints** (underspecification) and must clarify, and
  - hit **infeasible queries** (empty result set) once constraints are filled.
- Common behavior (“no results” / ad-hoc relaxations / fixed priority ordering) can violate user intent by dropping the *wrong* constraint.

## What is the core method / protocol?

- Defines **Preference-Aware Resolution**: when a query is infeasible, relax constraints in an order aligned to *user-specific* importance inferred from dialogue.
- Introduces **AWARE-US** (car-domain) where each instance:
  - is persona-grounded,
  - requires multi-turn clarification to elicit constraints,
  - is constructed to be infeasible under all constraints but feasible after dropping certain constraint(s).
- Dataset construction emphasizes controlled infeasibility via **minimal unsatisfiable subsets** (notably MUS-4).
- Three LLM-based importance estimators:
  1) **Local weighting**: infer a weight per constraint from the local clarification exchange.
  2) **Global one-shot weighting**: assign a normalized weight vector over constraints using the full transcript.
  3) **Pairwise ranking**: compare constraints pairwise to get an ordering.
- Outer loop is similar across methods: clarify → extract constraints → feasibility check → relax least-important → recommend from feasible set with preference-weighted scoring.

## What are the key metrics?

(From the paper’s framing; names vary by table/setting.)

- Dialogue completion / extraction quality (slot completion; #constraints parsed vs gold).
- Planning outcomes: feasibility after relaxation; recommendation rate; UNSAT rate.
- Oracle agreement (preference faithfulness):
  - **Relax match** (did the agent relax the oracle-designated least-important constraint?)
  - **Car match (gated)** (agreement on the final recommended item, typically conditioned on correct relaxation).

## What are the main results?

- Weighting-based methods improve **preference-faithful** infeasibility resolution relative to solver-style “minimal change” baselines.
- **Local weighting** is often best for end-to-end preference alignment (reported as higher oracle agreement; one headline number in the abstract is ~48% preference alignment in car recommendation).
- **Global weighting** can do best on “correct constraint relaxation” (~56% in the abstract) but may not translate as reliably to matching the oracle’s final recommendation.
- Pairwise ranking underperforms partly due to weaker constraint extraction / completion.

## How is this similar to GALILEO?

- Both care about **multi-turn interactions** where later behavior depends on earlier conversational signals.
- Shares a theme of **robustness under perturbations / failure modes** (here: empty-set / infeasibility rather than “answer flip” per se).
- Provides a concrete example of evaluating **trajectory-level correctness** instead of single-turn accuracy.

## How is this different from GALILEO?

- Domain/task: structured **tool-calling over a database** (car catalog) with explicit constraints vs GALILEO’s focus on broader conversational robustness phenomena.
- Supervision: uses **oracle relaxation targets / weights** induced by construction (MUS + sampled weights) rather than naturally-occurring human disagreement/pressure.
- The failure mode is **infeasibility** (empty results) and **constraint relaxation**, not belief change/stance change.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s evaluation targets real conversational robustness (e.g., persistence under pressure), it may cover more realistic “social” multi-turn failures than synthetic MUS-constructed constraint sets.

## Where GALILEO is weaker / needs to improve

- GALILEO writing could borrow this paper’s clarity in:
  - separating **dialogue-stage extraction** vs **downstream decision** errors,
  - reporting both *feasibility* and *preference-faithfulness* metrics.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider a GALILEO-style analysis split: “state/constraint extraction quality” vs “downstream policy/decision quality,” with metrics for each.
- [ ] If GALILEO has any “repair” step (e.g., after inconsistency), consider testing **local-vs-global weighting** analogs: per-turn vs transcript-level importance inference.
- [ ] Consider adopting an “oracle agreement” style metric (agreement with a defined target behavior) alongside outcome-based metrics.

## Quotes / details to potentially cite

- Problem framing: agents face “underspecification” and “infeasibility (empty set)” and ad-hoc relaxations can violate user intent.
- Benchmark summary (from abstract): “AWARE-US … 120+ persona-grounded queries … disambiguate … via conversation and … resolve infeasibility … consistent with persona-implied preferences.”
- Method summary (from abstract): three approaches: local weighting, global weighting, pairwise ranking.
