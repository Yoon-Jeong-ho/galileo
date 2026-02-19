# Ragas: Automated Evaluation of Retrieval Augmented Generation

- Year: 2023 (arXiv v1; updated v2 2025)
- Venue: arXiv (cs.CL)
- Authors: Shahul Es; Jithin James; Luis Espinosa-Anke; Steven Schockaert
- URL: https://arxiv.org/abs/2309.15217
- BibTeX key (if we add it): ragas2023
- Tags: rag, evaluation, llm-as-judge, faithfulness, context-relevance, reference-free

## One-sentence takeaway

RAGAS proposes a practical, reference-free evaluation suite for RAG pipelines that decomposes quality into faithfulness (grounding), answer relevance, and context relevance, computed via LLM prompting plus embeddings.

## What problem does it solve?

- RAG systems have multiple failure modes (bad retrieval, irrelevant/overlong context, ungrounded generation), but evaluation is often brittle: needs labeled QA pairs, relies on perplexity, or assumes access to token probabilities (not available for closed models).
- Need **fast, automated, reference-free** metrics to iterate on retrieval + generation choices in real-world RAG apps.

## What is the core method / protocol?

- Define a standard RAG setting: question q, retrieved context c(q), generated answer a(q).
- Provide **three main reference-free metrics**:
  - **Faithfulness**: whether claims in the answer are supported by the retrieved context.
    - Step 1: use an LLM to break the answer into atomic statements.
    - Step 2: for each statement, prompt an LLM to verify support from the context (Yes/No with brief rationale).
    - Score: fraction of statements judged supported.
  - **Answer relevance**: whether the answer addresses the question (penalize incomplete/off-topic/redundant content, irrespective of factuality).
    - Prompt LLM to generate multiple questions implied by the answer; embed them and compare to original question via cosine similarity; average similarity is the score.
    - (Paper uses OpenAI embeddings, e.g., text-embedding-ada-002 in the described implementation.)
  - **Context relevance**: whether the retrieved context contains only information needed to answer the question (penalize redundant/irrelevant context).
    - Prompt an LLM to extract the sentences in the context that are crucial for answering the question; compute a relevance ratio based on extracted vs total.
- Delivered as a framework/library (“ragas”), with integrations mentioned for LangChain and LlamaIndex.

## What are the key metrics?

- Faithfulness (statement support rate; LLM-verifier)
- Answer relevance (question-embedding similarity; LLM question generation + embedding model)
- Context relevance (LLM sentence extraction; relevance ratio)

## What are the main results?

- The paper positions RAGAS as a **developer-facing evaluation loop** enabling rapid comparisons of RAG variants without labeled ground truth.
- (Rapid review note: from the arXiv HTML excerpt read here, the detailed empirical correlation/benchmark numbers are not captured; consult the full paper/PDF for the quantitative results section.)

## How is this similar to GALILEO?

- Shares the core goal: **diagnose RAG pipeline quality** beyond end-task accuracy, especially around grounding/faithfulness.
- Uses decomposed dimensions (retrieval/context quality vs generation quality) rather than a single scalar score.

## How is this different from GALILEO?

- RAGAS is explicitly **LLM-as-judge + embedding-based**; it assumes access to a capable LLM for statement extraction/verification and for question generation.
- It is framed as a **reference-free** evaluation toolkit; GALILEO may target more structured/controlled evaluations, broader agentic behaviors, or different supervision signals (depending on our definition).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides deterministic checks, calibrated uncertainty, or evaluation with explicit evidence linking, it may be less sensitive to prompt/model choice than LLM-judge pipelines.
- If GALILEO avoids proprietary embedding/LLM dependencies, it may be more reproducible.

## Where GALILEO is weaker / needs to improve

- If we do not yet have an equivalent suite of **reference-free, pipeline-level diagnostics** (faithfulness/relevance/context focus), RAGAS highlights a pragmatic baseline users already adopt.
- Need to consider cost/latency trade-offs if we rely on LLM judges in evaluation loops.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite RAGAS as an early/popular **reference-free RAG evaluation** suite and contrast with our approach (what we measure differently, and why).
- [ ] Consider implementing a RAGAS-style faithfulness metric as a baseline comparator (statement decomposition + entailment-style verification).
- [ ] Evaluate judge-model sensitivity: how stable are scores across judge LLMs/prompts?
- [ ] Clarify in writing the distinction between “answer relevance” vs “faithfulness/grounding” (RAGAS offers a clean framing).

## Quotes / details to potentially cite

- RAGAS is “a framework for reference-free evaluation of Retrieval Augmented Generation (RAG) pipelines.”
- It targets multiple dimensions: retrieval finds “relevant and focused context passages,” LLM uses them “in a faithful way,” and overall “quality of the generation,” without “ground truth human annotations.”
- Faithfulness computation: decompose into statements, verify each against context with an LLM, score as |V|/|S|.
