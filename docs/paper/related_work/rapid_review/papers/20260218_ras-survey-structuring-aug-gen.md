# A Survey on Retrieval And Structuring Augmented Generation with Large Language Models

- Year: 2025
- Venue: KDD ’25 (Survey Track)
- Authors: Pengcheng Jiang, Siru Ouyang, Yizhu Jiao, Ming Zhong, Runchu Tian, Jiawei Han
- URL: https://arxiv.org/abs/2509.10697
- BibTeX key (if we add it): jiang2025_ras_survey
- Tags: rag, structuring, survey, retrieval

## One-sentence takeaway

A survey that frames “Retrieval And Structuring (RAS)” as RAG + explicit structure induction (taxonomies/hierarchies/IE/graphs) and reviews how structured representations are built and injected into LLM generation/reasoning.

## What problem does it solve?

- Vanilla RAG still retrieves mostly unstructured passages; these can be non-atomic/noisy, hard to verify, and weak for multi-hop or domain-organized queries.
- RAS aims to reduce hallucinations and improve grounding by (i) retrieving and (ii) organizing evidence into structured forms that are easier to integrate and reason over.

## What is the core method / protocol?

- Survey / taxonomy paper (not a new algorithm).
- Organizes the space along three axes:
  - Retrieval mechanisms: sparse, dense, hybrid retrieval.
  - Structuring techniques: taxonomy construction, hierarchical classification, information extraction (and broader “structured knowledge representations” like KGs).
  - Integration with LLMs: prompt-based injection, reasoning frameworks, and embedding/representation-based integration.
- Discusses challenges: retrieval efficiency, structure quality, and knowledge integration; and opportunities: multimodal retrieval, cross-lingual structures, interactive systems.

## What are the key metrics?

- Not a primary evaluation paper; metrics depend on the referenced RAG/RAS methods (typically QA accuracy, factuality/faithfulness, retrieval quality, efficiency/latency).

## What are the main results?

- Provides a unifying framing (“RAS”) and a map of method families + open problems; useful as a citation for motivation and terminology rather than empirical SOTA.

## How is this similar to GALILEO?

- Shares the motivation of improving reliability/grounding by connecting models to external information and enforcing better organization/structure.
- Potentially overlaps if GALILEO uses structured intermediate representations, decomposition, or evidence organization prior to generation.

## How is this different from GALILEO?

- Survey-level; does not propose a concrete evaluation protocol tightly aligned to model behavioral instability, drift, or intervention/recovery (GALILEO’s core focus).
- Focuses on retrieval/structuring pipelines rather than measuring conversational failure dynamics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO contributes a rigorous protocol/metric suite for robustness over time/turns, that is likely more targeted than this broad survey.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently assumes unstructured evidence, this survey suggests opportunities to add structure induction (hierarchies/taxonomies/IE/KG) as a stabilizing control.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite this survey to position “structured retrieval/organization” as a known lever beyond plain RAG.
- [ ] Consider an ablation: unstructured retrieval vs structured evidence representation (even lightweight: extracted entities/relations or hierarchical outlines) and measure impact on GALILEO’s robustness metrics.

## Quotes / details to potentially cite

- Defines RAS as integrating “dynamic information retrieval with structured knowledge representations” to address hallucination/outdated knowledge/limited domain expertise.
- Notes limits of traditional RAG: retrieved passages can contain “non-atomic information that can mislead LLMs” and struggle with complex multi-hop/domain organization (from the introduction in the HTML version).
