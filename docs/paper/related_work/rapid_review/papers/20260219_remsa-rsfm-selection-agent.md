# REMSA: An LLM Agent for Foundation Model Selection in Remote Sensing

- Year: 2025
- Venue: arXiv
- Authors: Binger Chen; Tacettin Emre Bök; Behnood Rasti; Volker Markl; Begüm Demir
- URL: https://arxiv.org/abs/2511.17442
- BibTeX key (if we add it): remsa2025
- Tags: remote-sensing, foundation-models, llm-agent, model-selection, metadata-database

## One-sentence takeaway

REMSA pairs a structured database of remote-sensing foundation-model metadata (150+ models) with an LLM agent that clarifies user constraints and ranks candidate models, outperforming retrieval/RAG-style baselines on an expert-verified query benchmark.

## What problem does it solve?

- Picking an appropriate remote-sensing foundation model (RSFM) for a task is hard because information is scattered across papers/model cards/repos, formats are heterogeneous, and deployment constraints (modalities, resolution, compute, etc.) are easy to miss.

## What is the core method / protocol?

- Build **RS-FMD**: a schema-guided database of RSFM metadata (claimed 150+ RSFMs; multi-modal inputs including SAR/multispectral/hyperspectral and vision-language).
- Build **REMSA (Remsa)**: an LLM-based agent that:
  - interprets a natural-language request,
  - asks/infers missing constraints (task, modality, resolution, compute, etc.),
  - retrieves candidates from RS-FMD,
  - ranks candidates using an agentic workflow with in-context learning,
  - produces a justification/trace for the recommendation.
- Evaluation setup: **75 expert-verified RS query scenarios**, expanded into **900 task-system-model configurations** under an “expert-centered” evaluation protocol.

## What are the key metrics?

- Expert-centered quality of the recommended model(s) for a given query scenario (details not in the abstract; likely agreement/acceptability vs expert expectations).
- Comparative performance vs baselines: naive agent, dense retrieval, and unstructured RAG-based LLM approaches.

## What are the main results?

- REMSA reportedly outperforms naive agent + dense retrieval + unstructured RAG baselines on the expert-verified scenario benchmark.
- System is explicitly constrained to **publicly available metadata** (no private data access).

## How is this similar to GALILEO?

- If GALILEO involves recommendation/selection under constraints (models/tools/strategies), REMSA is a close analog: **structured metadata + constraint clarification + transparent ranking**.
- Emphasizes reproducible decision support and justification, which maps to “auditability” style claims.

## How is this different from GALILEO?

- Domain-specific to remote-sensing foundation models and their metadata schema; less about generic long-horizon interaction dynamics.
- The central artifact is a **database + agentic selection workflow**, not a new foundation model or training method.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has a more controlled protocol separating different failure modes (e.g., pressure vs evidence), that would be a clearer causal story than “agent ranks better than baselines”.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a structured, schema-guided knowledge base of candidate capabilities/constraints, REMSA suggests this is a practical missing piece for reliability and reproducibility.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider a **schema-guided metadata layer** for whatever GALILEO selects/compares (models, tools, policies), so the agent reasoning is grounded in structured fields rather than free-form documents.
- [ ] Add a “constraint clarification” step to the protocol and evaluate it explicitly (does asking for missing constraints improve downstream selection quality?).
- [ ] If we need a small benchmark, mirror their pattern: a modest number of **expert-written query scenarios** expanded into many configurations.

## Quotes / details to potentially cite

- “RSFM Database (RS-FMD) … structured and schema-guided resource covering over 150 RSFMs …”
- “Remsa … the first LLM agent for automated RSFM selection from natural language queries … clarifies missing constraints, ranks models … provides transparent justifications.”
- “75 expert-verified RS query scenarios … 900 task-system-model configurations … expert-centered evaluation protocol.”
