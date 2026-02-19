# A Systematic Review of Key Retrieval-Augmented Generation (RAG) Systems: Progress, Gaps, and Future Directions

- Year: 2025
- Venue: arXiv (HTML version)
- Authors: Agada Joseph Oche; Ademola Glory Folashade; Tirthankar Ghosal; Arpan Biswas
- URL: https://arxiv.org/html/2507.18910v1
- BibTeX key (if we add it): Oche2025SystematicReviewRAG
- Tags: related-work, rag, survey, evaluation, enterprise, privacy-security

## One-sentence takeaway

A broad (mostly narrative) systematic review of RAG that organizes components, deployment issues, and recurring challenges, useful as a citation hub but not a source of new technical mechanisms.

## What problem does it solve?

- Consolidates a fast-growing body of Retrieval-Augmented Generation (RAG) work into an overview: foundations, components (retrieval / fusion / generation), year-by-year progress, enterprise deployment considerations, evaluation dimensions, and open challenges (retrieval quality, latency, privacy/security, integration overhead).

## What is the core method / protocol?

- “Systematic review” process:
  - Literature search across multiple libraries (ACL Anthology, IEEE Xplore, ACM DL, Google Scholar) for 2017–mid-2025.
  - Keyword-based queries around RAG / dense retrieval / hybrid retrieval / proprietary data / web search.
  - Inclusion/exclusion criteria focused on explicit retrieval+generation integration.
  - Synthesis grouped by year and by themes (foundations, enterprise, evaluation, challenges).
- Technical content is explanatory: defines canonical RAG formulation with latent-document marginalization (RAG-Sequence / RAG-Token), typical pipeline (chunking → embedding → retrieval → rerank → generation), and common retrievers (e.g., DPR) and fusion patterns.

## What are the key metrics?

- The paper frames evaluation dimensions rather than introducing a new benchmark:
  - Retrieval accuracy / relevance.
  - Generation quality / fluency.
  - Latency.
  - Computational efficiency.
  - (For enterprise) security/privacy handling and scalability.

## What are the main results?

- No new empirical result; main “results” are synthesized claims:
  - RAG mitigates hallucinations and enables knowledge updates without retraining by swapping/refreshing the corpus/index.
  - Two-stage retrieval (dense retrieve + cross-encoder rerank) is emphasized as a common recipe for better precision.
  - Enterprise RAG introduces additional constraints: proprietary data access controls, privacy, and integration complexity.
  - Persistent pain points: retrieval quality (recall/precision), privacy concerns, and pipeline overhead.

## How is this similar to GALILEO?

- If GALILEO is a retrieval-augmented/grounded generation system, this survey provides:
  - Canonical terminology (chunking, reranking, fusion strategies) and citations for baseline architectural choices.
  - A convenient “why RAG” motivation framing (hallucination mitigation, updating knowledge via non-parametric memory).

## How is this different from GALILEO?

- This is not a new model/system; it’s an organizing survey.
- It does not provide:
  - A concrete algorithmic contribution.
  - Reproducible experimental comparisons with standardized settings.
  - Detailed implementation guidance beyond common components.

## Where GALILEO is stronger / cleaner (if true)

- A focused system paper can be stronger than this survey by:
  - Providing a clear problem statement + measurable success criteria.
  - Giving end-to-end ablations (retrieval choice, chunking, reranking, fusion, prompt/control) under a single experimental protocol.
  - Reporting resource/latency tradeoffs with transparent setup.

## Where GALILEO is weaker / needs to improve

- If GALILEO’s related-work section lacks a broad positioning, this survey can be used to:
  - Support claims about common challenges (retrieval quality, privacy, integration overhead).
  - Provide citations for standard RAG formulations and variants (RAG-Sequence / RAG-Token) and “retrieval as latent variable” framing.

## Action items for GALILEO (experiments / method / writing)

- [ ] Use this survey as a secondary citation to justify: (a) why RAG, (b) common components (DPR + rerank), (c) enterprise constraints (privacy/security), but prioritize citing the original primary papers for any technical claim.
- [ ] When writing, consider adopting the survey’s evaluation dimensions as a checklist: retrieval metrics, generation metrics, latency/efficiency, plus privacy/security considerations if applicable.

## Quotes / details to potentially cite

- “Retrieval-Augmented Generation (RAG) represents a major advancement in natural language processing (NLP), combining large language models (LLMs) with information retrieval systems to enhance factual grounding, accuracy, and contextual relevance.” (Abstract)
- RAG motivation framing: mitigate hallucinations and outdated parametric knowledge; enable knowledge updates by modifying the corpus/index rather than retraining. (Intro)
- Canonical pipeline decomposition: chunking → embedding → (re)ranking → generation; and late-fusion marginalization variants (RAG-Sequence / RAG-Token) attributed to Lewis et al. (2020). (Foundations section)
