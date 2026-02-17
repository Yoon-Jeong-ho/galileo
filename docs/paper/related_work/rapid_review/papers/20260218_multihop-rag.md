# MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries

- Year: 2024
- Venue: arXiv
- Authors: Yixuan Tang, Yi Yang
- URL: https://arxiv.org/abs/2401.15391
- BibTeX key (if we add it): tang2024multihoprag
- Tags: rag, benchmark, multi-hop, evaluation

## One-sentence takeaway

MultiHop-RAG introduces a news-based multi-hop RAG benchmark (queries + gold answers + supporting evidence) and shows that common retrieval embeddings and strong LLMs still struggle on multi-hop evidence retrieval and reasoning.

## What problem does it solve?

- Existing RAG evaluations largely focus on single-hop retrieval (one piece of evidence) and do not isolate the harder setting where answering requires *multiple* supporting documents/facts.
- There was (per authors) no dedicated benchmark for multi-hop RAG that includes (i) a KB/corpus, (ii) multi-hop queries, (iii) ground-truth answers, and (iv) annotated supporting evidence.

## What is the core method / protocol?

- Build a benchmark dataset, **MultiHop-RAG**, using an **English news article dataset** as the underlying knowledge base.
- Provide:
  - a corpus/KB,
  - a large set of multi-hop queries,
  - ground-truth answers,
  - associated supporting evidence (multiple pieces).
- Benchmarking experiments (as described in abstract):
  1) Compare **embedding models** for retrieving evidence for multi-hop queries.
  2) Given retrieved evidence, compare **LLM reasoning/answering** ability across models (GPT-4, PaLM, Llama2-70B mentioned).

## What are the key metrics?

- Not specified in the abstract.
- Likely includes (a) evidence retrieval quality for multi-hop support and (b) answer correctness given evidence; confirm exact metrics if we later skim the PDF.

## What are the main results?

- Both retrieval (embedding-based evidence selection) and downstream answering (LLM reasoning over evidence) are **unsatisfactory** for multi-hop queries in their experiments.
- Even with strong LLMs (GPT-4 / PaLM / Llama2-70B named), multi-hop reasoning + evidence use remains a bottleneck.

## How is this similar to GALILEO?

- Shares the general theme of **robust evaluation**: building a targeted benchmark to expose failure modes that are not visible in simpler settings.
- The “multi-step / multi-piece” requirement is conceptually adjacent to multi-turn / multi-stage protocols (harder than single-shot tests).

## How is this different from GALILEO?

- Focuses on **information retrieval + multi-hop QA** for RAG, not social pressure / persuasion / multi-turn stance drift.
- Evaluates systems on evidence retrieval and reasoning, rather than measuring conversational instability dynamics (time-to-failure, flips, recovery).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s core claims are about multi-turn pressure dynamics and stability metrics, this benchmark does not address those dimensions.

## Where GALILEO is weaker / needs to improve

- If GALILEO needs more “task-realistic” settings, MultiHop-RAG is a reminder that **multi-step dependency** (needing multiple supports) can surface qualitatively different errors than single-hop tests.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding at least one evaluation slice where success requires *multiple* independent supports (multi-hop evidence), to ensure GALILEO claims generalize beyond single-evidence cases.
- [ ] If GALILEO discusses “drift vs evidence-driven revision,” consider adapting a multi-hop evidence setting as a cleaner *evidence* manipulation (multiple pieces added/removed).

## Quotes / details to potentially cite

- Abstract (dataset + findings): “MultiHop-RAG … consists of a knowledge base, a large collection of multi-hop queries, their ground-truth answers, and the associated supporting evidence.”
- Abstract (headline result): “Both experiments reveal that existing RAG methods perform unsatisfactorily in retrieving and answering multi-hop queries.”
