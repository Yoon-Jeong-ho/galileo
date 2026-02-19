# AI Agents Need Memory Control Over More Context

- Year: 2026
- Venue: arXiv
- Authors: Fouad Bousetouane (per arXiv submission metadata)
- URL: https://arxiv.org/abs/2601.11653
- BibTeX key (if we add it): bousetouane2026acc
- Tags: agent, memory, long-context, drift

## One-sentence takeaway

Agent Cognitive Compressor (ACC) replaces transcript replay with a bounded, turn-by-turn updated “compressed cognitive state” to reduce drift/hallucination in long-horizon agent workflows.

## What problem does it solve?

- Multi-turn agents degrade over long interactions due to (i) unbounded context growth (transcript replay), (ii) retrieval noise/selection errors, and (iii) “memory poisoning” where unverified information becomes persistent conditioning.
- Symptoms emphasized: loss of constraint focus, accumulation of errors, and memory-induced drift in operational workflows.

## What is the core method / protocol?

- Introduces **Agent Cognitive Compressor (ACC)**, a memory controller.
- Maintains a bounded internal state: **Compressed Cognitive State (CCS)**.
- Each turn updates CCS online using:
  - prior CCS,
  - current interaction/turn content,
  - (optionally) retrieved artifacts.
- Key design principle: **separate artifact recall from state commitment**:
  - retrieval can propose candidate facts/artifacts,
  - compression/controller decides what gets committed into CCS (to avoid unverified content ossifying into memory).
- Demonstrates how ACC plugs into common agent patterns (multi-turn ReAct; multi-turn planning).

## What are the key metrics?

- **Memory footprint / persistent context size** across turns (boundedness).
- **Task outcome / response quality** over multi-turn scenarios (judge-scored).
- **Memory-driven anomalies**, specifically:
  - hallucination rate (claim audit vs judge-maintained canonical state)
  - drift rate / constraint retention failures over turns.

## What are the main results?

- Across multi-turn operational scenarios (IT operations, cybersecurity response, healthcare workflows), ACC:
  - keeps **bounded memory** over long interactions,
  - shows **more stable behavior** across turns,
  - reduces **hallucination and drift** relative to transcript replay and retrieval-based memory baselines.
- Uses an **agent-judge-driven live evaluation** with turn-level scoring and bias controls (e.g., blinding / randomized presentation order).

## How is this similar to GALILEO?

- Shared focus: reliability in extended workflows where **constraints, entities, and intermediate decisions** must remain consistent.
- Treats memory as an explicit design axis (not just “more context”), and evaluates multi-turn stability (drift/constraint retention) rather than only single-turn accuracy.

## How is this different from GALILEO?

- ACC is positioned primarily as a **memory controller / cognitive compression** mechanism that replaces transcript replay, whereas GALILEO (as framed in our paper) may emphasize different core mechanisms (e.g., planning/control, verification, structured state, tool-use governance).
- ACC’s key abstraction is a single bounded CCS plus artifact store, with an explicit “commit” step; GALILEO may use different modularization (e.g., separate world state, task state, policies, or proofs/constraints).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses more explicit structure/verification for state updates (e.g., typed state, provenance, constraints-as-code), we can claim clearer guarantees than purely text-compressed state.
- If GALILEO already has a notion of “confirmed vs unconfirmed” information (or explicit gating), we can position it as a principled instantiation of ACC’s “commitment” idea.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently relies on transcript replay / ad-hoc summarization, this paper is a strong prompt to formalize **bounded working state** and **commit gating**.
- We may need better evaluation for **drift and memory-driven hallucinations** across many turns.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add/strengthen a **memory-control framing**: “bounded working state” + “artifact recall vs state commitment”.
- [ ] Consider implementing a **commit gate** for long-horizon runs: only verified/required invariants enter persistent state.
- [ ] Add evaluation metrics similar to this paper: **constraint retention / drift**, and **claim-audit hallucination** over turns.
- [ ] Add an operational multi-turn benchmark slice (or internal scenarios) with **injected distractions** and evolving constraints.

## Quotes / details to potentially cite

- Problem framing (from abstract): long workflows degrade due to “loss of constraint focus, error accumulation, and memory-induced drift.”
- Critique of baselines: transcript replay causes unbounded growth; retrieval is vulnerable to noisy recall and “memory poisoning.”
- Method line: ACC “replaces transcript replay with a bounded internal state updated online at each turn” and “separates artifact recall from state commitment.”
- Evaluation: “agent-judge-driven live evaluation framework” measuring outcomes plus “memory-driven anomalies” (hallucination and drift) across extended interactions.
