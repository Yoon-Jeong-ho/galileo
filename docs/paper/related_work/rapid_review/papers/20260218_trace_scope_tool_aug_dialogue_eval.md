# Multi-Faceted Evaluation of Tool-Augmented Dialogue Systems (TRACE/SCOPE)

- Year: 2025
- Venue: arXiv (manuscript under submission)
- Authors: Zhaoyi Joey Hou; Tanya Shourya; Yingfan Wang; Shamik Roy; Vinayshekhar Bannihatti Kumar; Rashmi Gangadharaiah
- URL: https://arxiv.org/abs/2510.19186
- BibTeX key (if we add it): trace_scope_2025
- Tags: tool-use, multi-turn, evaluation, benchmarks, user-satisfaction-vs-correctness

## One-sentence takeaway

Proposes a tool-augmented dialogue benchmark with systematic *negative/error* cases (TRACE) and an evaluation framework (SCOPE) meant to catch failures that look “user-satisfying” but are tool-incorrect or agent-inappropriate.

## What problem does it solve?

- Existing tool-use benchmarks/evaluators often (i) focus on tool-call correctness alone, or (ii) focus on user satisfaction/conversation quality, missing important multi-turn failure modes.
- Key gap highlighted: **users can appear satisfied even when the agent misused/misread tools**, fabricated tool outcomes, or failed silently—so user-satisfaction-only signals are insufficient.

## What is the core method / protocol?

- **TRACE benchmark**: synthetic multi-turn tool-augmented conversations covering both successful and erroneous tool use.
  - They define 4 dimensions to characterize scenarios:
    - tool execution correctness (correct / incorrect due to agent / incorrect due to tool error)
    - agent performance (appropriate vs inappropriate response to the situation)
    - user satisfaction (satisfied vs dissatisfied)
    - overall conversation success label (POS/NEG), combining the above beyond surface satisfaction
  - They derive **26 distinct “situations”** from plausible combinations and generate conversations per situation.
  - Tool set: selection of tools drawn from prior tool-use resources (e.g., ToolTalk, MINT, API-Bank), standardized into schemas.
  - They generate conversations with LLM prompting; tool executions are *simulated* in generation.
  - Quality control via a tuned LLM-judge filter (high precision emphasis), plus a smaller human-filtered “gold” subset.

- **SCOPE evaluation framework**: aims to “automatically discover” evaluation areas/rubrics and incorporate tool-specific error categories when scoring tool-augmented dialogues.
  - Compared against a user-satisfaction-centered baseline (SPUR).

## What are the key metrics?

- Primary evaluation appears to be **accuracy** at identifying/labeling conversation quality/failure modes on TRACE (POS vs NEG and/or related labels).
- Emphasis case: “challenging” subsets where **user satisfaction is misleading**.

## What are the main results?

- TRACE size reported in the paper HTML: **516 conversations** total (182 POS / 334 NEG), with intentionally more NEG to cover diverse error cases.
- SCOPE reportedly improves over the baseline, with headline improvements up to **+17.6% accuracy**, and much larger gains (reported up to **+47% accuracy**) on cases where user satisfaction signals are misleading.

## How is this similar to GALILEO?

- Shares the core motivation that **single scalar evaluations can be misleading** in multi-turn settings (e.g., surface-level “looks fine” vs underlying failure).
- Aligns with the idea that evaluation should separate *observable conversation smoothness* from *latent correctness/faithfulness* (in their case: tool correctness + agent behavior).

## How is this different from GALILEO?

- Focuses on **tool-augmented dialogue correctness and tool-error handling**, not (primarily) belief drift / persuasion / pressure robustness.
- TRACE is largely **synthetic** (LLM-generated dialogues with simulated tool runs), whereas GALILEO’s emphasis is likely on more direct multi-turn robustness constructs (e.g., drift vs revision controls, recovery trajectories).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides explicit *pressure-only vs evidence-bearing* controls, and/or *recovery-after-failure* trajectory metrics, it likely offers a cleaner causal story than “overall success” labels.

## Where GALILEO is weaker / needs to improve

- If GALILEO includes any tool-augmented agent setting, we may currently lack a crisp taxonomy for **tool error classes** (agent-parameter errors vs tool failures) and how these interact with user satisfaction.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite TRACE/SCOPE as evidence that **user-satisfaction-style judges miss subtle but important multi-turn errors** (surface OK, latent wrong).
- [ ] Consider adding an ablation/analysis section explicitly separating “looks good to user” vs “is actually correct” (even without tools): e.g., *apparent compliance* vs *truth/consistency*.
- [ ] If we evaluate agents with tools, borrow their decomposition: (tool correctness) × (agent response appropriateness) × (user satisfaction) and report mismatched quadrants.

## Quotes / details to potentially cite

- Problem framing: errors in tool-augmented dialogues can come from complex user–agent–tool interactions; user satisfaction can be misleading when the agent misinterprets tool outputs.
- TRACE construction: 4 key dimensions; 26 situations; NEG-heavy benchmark to cover diverse error cases.
- Results: SCOPE improves accuracy over SPUR, especially on misleading-satisfaction cases.
