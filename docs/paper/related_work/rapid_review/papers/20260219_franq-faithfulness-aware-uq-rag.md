# Faithfulness-Aware Uncertainty Quantification for Fact-Checking the Output of Retrieval Augmented Generation

- Year: 2025
- Venue: arXiv (cs.CL)
- Authors: Ekaterina Fadeeva; Aleksandr Rubashevskii; Dzianis Piatrashyn; Roman Vashurin; Shehzaad Dhuliawala; Artem Shelmanov; Timothy Baldwin; Preslav Nakov; Mrinmaya Sachan; Maxim Panov
- URL: https://arxiv.org/abs/2505.21072
- BibTeX key (if we add it): fadeeva2025franq
- Tags: rag, hallucination-detection, uncertainty-quantification, faithfulness, factuality, fact-checking

## One-sentence takeaway

FRANQ improves RAG hallucination detection by explicitly separating *faithfulness-to-retrieved-evidence* from *objective factuality*, and applying different uncertainty signals depending on whether a claim is entailed by retrieval.

## What problem does it solve?

- In RAG, many detectors/benchmarks treat “not supported by retrieved context” as “hallucination”, conflating:
  - **faithfulness** (entailed by retrieved passages) vs
  - **factuality** (true in the world).
- This can mislabel correct statements that are simply *not present* in retrieved snippets, and it obscures two different failure modes:
  - retrieval-grounded but wrong (retrieval itself wrong / misused), vs
  - model-internal-knowledge errors.

## What is the core method / protocol?

- Decompose a long-form RAG answer into **atomic claims**.
- For each claim, estimate:
  1) **P(claim is faithful to retrieval r)**
  2) **P(claim is true | faithful)**
  3) **P(claim is true | unfaithful)**
- Combine via law of total probability:
  - P(true) = P(faithful)*P(true|faithful) + (1-P(faithful))*P(true|unfaithful)
- In practice this is “faithfulness-aware UQ”: run different uncertainty quantification strategies depending on whether the claim is supported/entailed by retrieved context.
- The paper positions FRANQ against a set of UQ baselines applied directly on the joint prompt (query + retrieval), including:
  - information-based: max prob / perplexity / token entropy / claim-conditioned probability (CCP)
  - reflexive: P(True) prompting
  - sample-diversity: semantic entropy, lexical similarity, spectral measures (e.g., degree matrix / eigenvalues)

## What are the key metrics?

- Hallucination / factual-error detection performance at the **claim level** for long-form QA (and response-level for short-form QA).
- Uses datasets with **both factuality and faithfulness labels**; long-form setting emphasizes multi-claim answers.

## What are the main results?

- Across multiple datasets / tasks / LLMs, FRANQ reports better detection of factual errors in RAG outputs than applying standard UQ methods naively on the combined (query + retrieval) prompt.
- A key qualitative claim: conditioning uncertainty on faithfulness improves calibration for the two different error sources (retrieval-grounded vs model-internal).

## How is this similar to GALILEO?

- Same overall theme: **reliability estimation for LLM outputs** via uncertainty / self-evaluation signals.
- Similar decomposition mindset: break an output into units (claims) and score them, rather than treating the whole answer monolithically.

## How is this different from GALILEO?

- FRANQ is RAG-specific and explicitly models a **two-input structure** (query vs retrieved evidence) and their mismatch.
- Their central move is *faithfulness-first routing* of UQ; GALILEO may be aiming at broader agentic behaviors or general-purpose reliability without requiring retrieval.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets general agent outputs, tool-augmented actions, or non-RAG settings, it can cover broader failure modes than retrieval-anchored QA.
- If GALILEO avoids needing claim extraction and entailment-style faithfulness checks, it may be operationally simpler.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently uses a single “support” notion, FRANQ is a reminder to **separate truth from evidence-coverage**.
- If GALILEO doesn’t explicitly represent “evidence present but wrong” vs “no evidence but true”, it may misclassify similar cases.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, explicitly call out the **factuality vs faithfulness** distinction, and why faithfulness-only evaluation can be misleading.
- [ ] Consider a GALILEO ablation that routes reliability scoring based on a preliminary **evidence-entailment/faithfulness** decision (even outside RAG, this can map to “tool logs / citations / observations”).
- [ ] If claim-level scoring is part of GALILEO, compare against FRANQ-style decomposition: P(true)=P(grounded)*P(true|grounded)+...

## Quotes / details to potentially cite

- “Existing approaches … often conflate factuality with faithfulness to the retrieved evidence, incorrectly labeling factually correct statements as hallucinations if they are not explicitly supported by the retrieval.” (abstract)
- FRANQ defines faithfulness as entailment by retrieved context, and factuality as objective correctness; then conditions UQ on faithfulness.
