# A Taxonomy of Prompt Defects in LLM Systems

- Year: 2025
- Venue: arXiv
- Authors: Haoye Tian et al.
- URL: https://arxiv.org/abs/2509.14404
- BibTeX key (if we add it): tian2025taxonomy_prompt_defects
- Tags: prompt-engineering, taxonomy, defects, robustness, software-engineering

## One-sentence takeaway

A software-engineering-grounded taxonomy of recurring **prompt defects** (and mitigations) that cause LLM systems to behave unreliably, insecurely, or inefficiently.

## What problem does it solve?

- Prompting is the “programming interface” for many LLM systems, but failures are often treated as ad-hoc; the paper aims to systematize *how* prompts fail and how to mitigate those failures.

## What is the core method / protocol?

- Survey + taxonomy: organizes prompt defects along **six dimensions**:
  1) Specification & intent
  2) Input & content
  3) Structure & formatting
  4) Context & memory
  5) Performance & efficiency
  6) Maintainability & engineering
- Breaks each dimension into finer subtypes, with examples + root-cause analysis.
- Maps defect → impact → mitigation strategies (prompt patterns, guardrails, testing/eval frameworks).

## What are the key metrics?

- Not a benchmark paper; no single quantitative metric is central (primarily a conceptual/engineering taxonomy).

## What are the main results?

- Delivers a “master taxonomy” linking common defect subtypes to downstream impacts and mitigation strategies.
- Highlights that prompt failures show up as reliability/security/efficiency issues in real workflows, motivating more rigorous engineering methods (testing harnesses, evaluations, guardrails).

## How is this similar to GALILEO?

- Overlaps in motivation: **dependability of LLM-driven systems** and avoiding “small prompt mistakes → large downstream failures.”
- Provides vocabulary that could help describe/organize *failure modes* that GALILEO’s evaluations may surface (especially around context/memory and spec ambiguity).

## How is this different from GALILEO?

- Focuses on prompt engineering defects broadly (single- and multi-turn), not specifically on GALILEO’s core phenomena (multi-turn social pressure / drift-vs-revision / recovery dynamics).
- Primarily descriptive taxonomy rather than a new evaluation protocol with quantitative results.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes controlled multi-turn protocols + metrics (e.g., time-to-failure / recovery), it offers a more *operationalized* evaluation story than a taxonomy.

## Where GALILEO is weaker / needs to improve

- GALILEO writing may benefit from an explicit mapping from observed failures to a standardized “defect” vocabulary (spec/format/context/memory/maintainability), to help readers connect results to engineering remedies.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short related-work paragraph framing GALILEO’s stress tests as surfacing a subset of “prompt defects” (esp. spec ambiguity + context/memory) and positioning our contributions as *measurement + controlled protocols*.
- [ ] Consider a small appendix table: **GALILEO failure mode → (closest prompt-defect dimension) → mitigation knob** (guardrail, prompt pattern, test).

## Quotes / details to potentially cite

- Abstract-level positioning: prompts as de-facto programming interface; “small mistakes can cascade into unreliable, insecure, or inefficient behavior.”
- The six dimensions listed in the abstract (Specification/Intent; Input/Content; Structure/Formatting; Context/Memory; Performance/Efficiency; Maintainability/Engineering).
