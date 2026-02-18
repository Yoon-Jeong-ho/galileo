# Arbor: A Framework for Reliable Navigation of Critical Conversation Flows

- Year: 2026
- Venue: arXiv
- Authors: Luís Silva, Diogo Gonçalves, Catarina Farinha, Clara Matos, Luís Ungaro
- URL: https://arxiv.org/abs/2602.14643
- BibTeX key (if we add it): arbor2026
- Tags: multi-turn, conversation-flows, workflow-adherence, orchestration, retrieval, decomposition, clinical-triage

## One-sentence takeaway

Arbor improves multi-turn workflow adherence in decision-tree-guided conversations by decomposing “where to go next?” (transition selection) from “what to say?” (response generation) and retrieving only local node context each turn.

## What problem does it solve?

- In high-stakes structured workflows (e.g., clinical triage decision trees), LLMs embedded in a single giant prompt tend to drift: they lose track of state, miss relevant parts of long prompts (lost-in-the-middle), and violate the intended decision logic as the tree/prompt grows.
- “Monolithic prompting” also mixes heterogeneous subtasks (state tracking, tree parsing, transition evaluation, natural language generation) in one call, making failures hard to debug.

## What is the core method / protocol?

- Represent the decision tree as a standardized edge-list, stored externally for dynamic retrieval.
- Use a DAG-style orchestration loop over turns:
  - Retrieve only the outgoing edges of the *current* node (local neighborhood) instead of the full tree.
  - Make a dedicated LLM call to evaluate which transition is valid / should be taken given the conversation state/user message.
  - Make a separate LLM call to generate the user-facing response conditioned on the selected next node (separation of concerns).
- Key design claim: architectural decomposition + local retrieval reduces dependence on “intrinsic” model capability and context-length robustness.

## What are the key metrics?

- Turn-level accuracy (selecting the correct transition / node for each annotated turn).
- Efficiency: per-turn latency and per-turn cost.
- (Implied) overall path/outcome correctness along the decision tree, though the headline numbers are reported at turn level.

## What are the main results?

- Compared to a single-prompt baseline (full decision tree embedded in the prompt), across 10 foundation models on annotated turns from real clinical triage conversations:
  - +29.4 percentage points mean turn accuracy.
  - -57.1% per-turn latency.
  - 14.4× lower per-turn cost (average).
- Smaller models under Arbor can match or exceed larger models under monolithic prompting.

## How is this similar to GALILEO?

- Same general theme: multi-turn robustness is often an *architecture/orchestration* problem, not just a base-model capability problem.
- Uses decomposition into substeps and “retrieve only what matters now” to avoid long-context brittleness.
- Emphasizes auditability/debuggability by isolating failure modes to specific subcalls (transition choice vs response).

## How is this different from GALILEO?

- Target task is structured workflow adherence (decision-tree navigation) in a high-stakes domain, rather than (primarily) belief/stance stability or social-pressure robustness.
- The “ground truth” is typically a known decision-tree path, not an epistemic stance truthfulness measure.
- Evaluates on annotated clinical triage conversation turns; domain constraints/ethics may limit data release and generalization.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is focused on measuring/mitigating *sycophancy / pressure-induced flips* or *belief drift*, it likely has a cleaner conceptual separation of “agreement bias” vs “workflow adherence”.
- GALILEO-style stress testing could cover broader interaction types beyond deterministic trees.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently relies on long prompts or monolithic agent loops for multi-step protocols, Arbor suggests a concrete alternative: node-local retrieval + specialized calls.
- If GALILEO’s evaluations don’t report latency/cost tradeoffs, Arbor shows these can be major wins from decomposition.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding an “orchestration ablation” section: monolithic prompt vs decomposed multi-call pipeline (transition-selection vs response-generation) to quantify robustness gains.
- [ ] Add a failure taxonomy aligned with separation-of-concerns (state tracking vs transition selection vs response quality) to improve debuggability.
- [ ] If applicable, explore representing GALILEO’s multi-turn protocol as an explicit graph/DAG with local retrieval, to reduce context-length dependence.
- [ ] When positioning, cite Arbor as evidence that architectural decomposition can yield large robustness + efficiency improvements in multi-turn, high-stakes workflows.

## Quotes / details to potentially cite

- “Monolithic approaches that encode entire decision structures within a single prompt are prone to instruction-following degradation as prompt length increases, including lost-in-the-middle effects and context window overflow.”
- “At runtime, a directed acyclic graph (DAG)-based orchestration mechanism iteratively retrieves only the outgoing edges of the current node, evaluates valid transitions via a dedicated LLM call, and delegates response generation to a separate inference step.”
- “Arbor improves mean turn accuracy by 29.4 percentage points, reduces per-turn latency by 57.1%, and achieves an average 14.4× reduction in per-turn cost.”
