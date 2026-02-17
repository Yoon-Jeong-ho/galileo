# PersistBench: When Should Long-Term Memories Be Forgotten by LLMs?

- Year: 2026
- Venue: arXiv
- Authors: Sidharth Pulipaka, Oliver Chen, Manas Sharma, Taaha S Bajwa, Vyas Raina, Ivaxi Sheth
- URL: https://arxiv.org/abs/2602.01146
- BibTeX key (if we add it): persistbench_pulipaka_2026
- Tags: agents, long-term-memory, safety, context-leakage, sycophancy, benchmark

## One-sentence takeaway

PersistBench benchmarks two long-term-memory-specific safety failure modes—cross-domain leakage and memory-induced sycophancy—and finds very high failure rates across 18 LLMs, motivating “use memory vs forget memory” mechanisms.

## What problem does it solve?

- As production assistants increasingly inject persistent user memories into the system context, we lack evaluations of *safety harms caused by that persistence* (not just “can it remember?”).
- The paper argues two key risks are under-measured:
  - **Cross-domain leakage:** irrelevant memories injected into unrelated tasks degrade/derail responses.
  - **Memory-induced sycophancy:** stored user attributes/beliefs amplify agreement/echo-chamber behavior.

## What is the core method / protocol?

- Construct **PersistBench**: (memory set, query) pairs designed to probe:
  - cross-domain leakage,
  - memory-induced sycophancy,
  - plus a **beneficial memory** set to ensure mitigations don’t just “turn memory off”.
- Evaluate **18 frontier + open-weight** LLMs.
- Failure is defined at the *response level* (ranging from mild irrelevant recall to visibly derailed/harmful outputs), intended to reflect end-user harm.

## What are the key metrics?

- **Failure rate** on:
  - cross-domain leakage samples
  - sycophancy samples
  - beneficial memory samples (as a counterbalance)

## What are the main results?

- Median failure rate reported in the abstract:
  - **53%** on cross-domain leakage samples.
  - **97%** on memory-induced sycophancy samples.
- The introduction notes:
  - identity-validation-type samples are especially sycophancy-inducing.
  - “beneficial memory” performance is only weakly correlated with safety performance (suggesting naive suppression is not a clean fix).

## How is this similar to GALILEO?

- Shares the core concern that **state/history (here: persistent memory) can systematically distort behavior** over time, producing drift-like failure modes and misalignment with objective truth.
- Offers a concrete benchmark framing for *long-horizon, multi-session* effects that GALILEO-style evaluations can connect to.

## How is this different from GALILEO?

- PersistBench is specifically about **long-term memory injection across sessions** (often as static text in system prompt), rather than within-session multi-turn dynamics alone.
- The two primary labels are **leakage** and **sycophancy**; it is less focused on fine-grained trajectory metrics (e.g., survival/time-to-failure) and more on response-level safety failures.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO emphasizes *within-dialogue controllability/measurement* (e.g., explicit interventions, time-to-failure), it may offer cleaner causal attribution than “static memory injected upfront”.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not explicitly model **persistent user-profile memory** across sessions, it may miss a major real-world deployment vector for multi-turn/long-horizon failures.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “persistent memory” condition: inject a compact memory block into system context and evaluate (a) cross-domain leakage and (b) agreement/echo-chamber effects.
- [ ] Define a GALILEO-compatible metric for leakage/derailment (e.g., task-relevance scoring + refusal correctness) and compare with PersistBench failure rate.
- [ ] In related work: position PersistBench as *memory-specific* safety benchmark adjacent to multi-turn sycophancy/pressure evaluations.

## Quotes / details to potentially cite

- Abstract (two risks): “cross-domain leakage” and “memory-induced sycophancy”.
- Abstract (headline failures): “a median failure rate of **53%** on cross-domain samples and **97%** on sycophancy samples.”
