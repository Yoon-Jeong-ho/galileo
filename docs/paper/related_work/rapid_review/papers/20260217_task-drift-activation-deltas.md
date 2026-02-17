# Get my drift? Catching LLM Task Drift with Activation Deltas

- Year: 2024
- Venue: SaTML 2025 (arXiv v6)
- Authors: Sahar Abdelnabi, Aideen Fay, Giovanni Cherubin, Ahmed Salem, Mario Fritz, Andrew Paverd
- URL: https://arxiv.org/abs/2406.00799
- BibTeX key (if we add it): abdelnabi2024get
- Tags: drift, detection, activations, RAG, prompt-injection, monitoring

## One-sentence takeaway

Shows that **activation deltas** (hidden-state differences before vs after ingesting external context) enable a simple linear probe to detect **task drift** from prompt-injection-like inputs with near-perfect AUC, and releases a large-scale TaskTracker toolkit.

## What problem does it solve?

- In retrieval-augmented / tool-augmented apps, external text (search results, emails, docs) can embed *instructions* that the model follows, deviating from the user’s intended task.
- The paper defines this deviation as **task drift** and frames it as a security/reliability issue (data treated as executable).

## What is the core method / protocol?

- Compute **activation deltas**: difference between model activations *before* and *after* processing external data added to the context.
- Train simple probes (incl. linear classifiers) to predict whether drift occurred.
- Evaluation emphasizes robustness: minimal assumptions about phrasing of tasks/system prompts/attacks; tests out-of-distribution.
- Releases **TaskTracker**: dataset (>500K instances), representations from six SOTA LMs, and inspection tooling.

## What are the key metrics?

- Drift detection performance: **ROC AUC** (reported as near-perfect on OOD tests).
- (Implied) generalization across unseen task domains / attack types.

## What are the main results?

- Activation deltas are **strongly correlated** with task drift.
- A **simple linear classifier** can detect drift with **near-perfect ROC AUC** even on OOD test sets.
- Generalizes to unseen domains including **prompt injections, jailbreaks, and malicious instructions** without being trained on those attacks.
- No need to modify the base model (no fine-tuning required), so it is deployable alongside meta-prompting defenses.

## How is this similar to GALILEO?

- Shared theme: **multi-turn / contextual degradation** of intended behavior (“drift”) due to interaction history or injected content.
- Aligns with a *monitoring* narrative: detect when the model is leaving the intended trajectory.

## How is this different from GALILEO?

- Focus is **RAG / prompt-injection task drift**, not social-pressure persuasion or belief/answer flip dynamics.
- Uses **internal activations** and probes (not black-box metrics), which may be infeasible for closed models.
- Drift event is tied to ingesting external context, rather than gradual multi-turn pressure.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is black-box: can evaluate across closed APIs without hidden-state access.
- If GALILEO targets social-pressure robustness: addresses a different failure mode than prompt injection.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently lacks *early-warning monitors*, this paper is strong evidence that **state-based drift detectors** can work very well (when activations are accessible).

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a brief “monitoring/inspection” related-work paragraph: activation-based drift detection complements outcome-only benchmarks.
- [ ] If we have access to hidden states in any open-weight models, consider a small side experiment: can activation deltas predict impending **pressure-induced flips** (time-to-event risk monitoring analog)?

## Quotes / details to potentially cite

- Defines deviation induced by external-data instructions as **“task drift.”**
- Activation deltas: “the difference in activations before and after processing external data” used to detect drift.
- Releases TaskTracker toolkit with **>500K instances** and representations from **six** models.
