# HalluHard: A Hard Multi-Turn Hallucination Benchmark

- Year: 2026
- Venue: arXiv (submitted under “Machine Learning, ICML” track label on the HTML page)
- Authors: Dongyang Fan et al.
- URL: https://arxiv.org/abs/2602.01031
- BibTeX key (if we add it): halluhard2026fan
- Tags: hallucination, multi-turn, benchmark, groundedness, citation-checking, web-search, evaluation

## One-sentence takeaway

HalluHard is a 950-item multi-turn benchmark that forces *inline citations* and uses a web-search-based judge that reads full-text sources (incl. PDFs) to separate “bad citation” vs “unsupported content,” showing even frontier models still hallucinate at high rates (~30% with web search in their strongest setting).

## What problem does it solve?

- Existing hallucination benchmarks often saturate and/or are single-turn and format-constrained, under-representing real multi-turn workflows where early mistakes propagate.
- In open-ended answers, “having a plausible citation” is not enough: models can cite an ostensibly relevant source while still fabricating details the source does not support.

## What is the core method / protocol?

- **Benchmark:** 950 “seed questions” spanning 4 high-stakes domains:
  - legal cases
  - research questions
  - medical guidelines
  - coding
- **Generation constraint:** the model must include **inline citations** for factual assertions.
- **Judge pipeline:** an iterative, tool-using judging procedure that:
  - extracts/checks claims,
  - retrieves evidence via web search,
  - fetches + filters + parses full text sources (including PDFs),
  - returns structured verdicts distinguishing (at least) **reference-grounding** vs **content-grounding** failures.

## What are the key metrics?

- **Hallucination rate** on multi-turn conversations (aggregate; reported with/without web search).
- Error-type decomposition: **reference grounding** (wrong/irrelevant citation) vs **content grounding** (citation exists but doesn’t support the claimed details).
- Analyses by **turn position**, **model capacity**, **reasoning effort/effective thinking**, and **knowledge type/domain**.

## What are the main results?

- Even with web search enabled, hallucinations remain substantial; the abstract reports ~**30%** hallucination rate in the strongest configuration they tested.
- Without web search, hallucination rates are much higher (the intro figure description mentions ~**60%** without WS for the same frontier model).
- Hallucinations tend to **increase in later turns**, consistent with error propagation/cascading.
- “Thinking”/reasoning can reduce hallucinations, but more reasoning effort does not necessarily monotonically improve outcomes (per their summary claims).

## How is this similar to GALILEO?

- Shared motivation: **multi-turn reliability degrades over turns**; early errors can cascade.
- Emphasizes **trajectory-level evaluation** (turn position as a driver), aligning with GALILEO-style “when does failure happen?” framing.

## How is this different from GALILEO?

- Target failure mode is **hallucination/groundedness** rather than social-pressure-driven drift/sycophancy (though both are multi-turn reliability failures).
- Requires **inline citations** and builds evaluation around **evidence retrieval + full-text verification**, whereas GALILEO is more about belief/stance dynamics under interventions/pressure.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has explicit controls for **pressure-driven drift vs evidence-driven revision**, that’s a cleaner causal framing than a single “hallucination” label.
- If GALILEO doesn’t rely on web-scale retrieval during evaluation, it may offer a simpler-to-run protocol; HalluHard’s judge pipeline is tool-heavy.

## Where GALILEO is weaker / needs to improve

- If GALILEO evaluates factual claims, HalluHard’s **content-grounding check** highlights that “citation present” (or “sounds plausible”) is insufficient without source-reading verification.
- Multi-domain coverage (legal/medical/coding) is a strong “high-stakes realism” argument; GALILEO may need at least one comparable slice or a clearer generality argument.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a *citation-required* variant (even for a small subset) to audit grounding failures separately from stance drift.
- [ ] Borrow the **reference-grounding vs content-grounding** decomposition as an analogy for “appears calibrated/justified vs actually supported.”
- [ ] If we use any retrieval/judging, explicitly document whether we verify **full-text support** (not just URL relevance), to avoid the “good citation, fabricated details” trap.

## Quotes / details to potentially cite

- “We introduce HalluHard, a challenging multi-turn hallucination benchmark with 950 seed questions spanning four high-stakes domains: legal cases, research questions, medical guidelines, and coding.”
- “We operationalize groundedness by requiring inline citations for factual assertions.”
- “Across a diverse set of frontier proprietary and open-weight models, hallucinations remain substantial even with web search (≈30% for the strongest configuration…).”
