# Decoupled Reasoning with Implicit Fact Tokens (DRIFT): A Dual-Model Framework for Efficient Long-Context Inference

- Year: 2026
- Venue: arXiv
- Authors: Wenxuan Xie, Yujia Wang, Xin Tan, Chaochao Lu, Xia Hu, Xuhong Wang
- URL: https://arxiv.org/abs/2602.10021
- BibTeX key (if we add it): Xie2026DRIFT
- Tags: long-context, prompt-compression, dual-model, latent-tokens, efficiency, robustness-adjacent

## One-sentence takeaway

DRIFT separates (query-conditioned) knowledge compression from reasoning by having a small “knowledge model” turn long documents into dense implicit fact tokens that a larger reasoning model can consume efficiently, improving long-context QA accuracy and latency.

## What problem does it solve?

- Long-context / knowledge-intensive inference is expensive and error-prone when feeding raw documents (quadratic attention cost, context limits, redundancy).
- RAG has retriever noise/ceilings; parametric editing risks forgetting; static prompt compression can drop query-critical evidence.

## What is the core method / protocol?

- Dual-model architecture:
  - **Knowledge model (lightweight):** splits documents into chunks and produces **implicit fact tokens** *conditioned on the query* (dynamic compression rather than query-agnostic summarization).
  - **Projection step:** maps these dense tokens into the **reasoning model’s embedding space**.
  - **Reasoning model (larger):** performs inference using projected fact tokens instead of raw text.
- Emphasis on **information-adaptive compression** (identify/abstract core info before encoding), and discussion of going “beyond fixed-ratio” compression (bucketed / adaptive token budgeting).

## What are the key metrics?

- Task accuracy on long-context benchmarks (e.g., LongBench v2 reported in intro).
- Compression ratio (e.g., 32×, 64×, 128×).
- Inference latency / speedup (reported in intro).

## What are the main results?

- Reported (intro): ~**7× speedup** on **256k-token** documents (average).
- Reported (intro): with a Mistral-7B-based setup, **32× compression** while improving **LongBench v2** accuracy from **20.87% → 29.22%**.
- Reported (intro): remains competitive even at **64× / 128×** compression, suggesting robustness under extreme compression.

## How is this similar to GALILEO?

- Conceptual neighbor on **decoupling**: separate modules for “retain/extract relevant information” vs “reason/infer”, which is relevant to long-horizon agent settings.
- Shares the motivation of avoiding brittle pipelines (e.g., pure retrieval ceilings) by learning intermediate representations.

## How is this different from GALILEO?

- Focus is **single-query long-context inference** (document QA / long-context benchmarks), not **multi-turn interaction dynamics** (drift, sycophancy, belief change) that GALILEO targets.
- “DRIFT” here is a *method name*; it is not about behavioral drift across turns.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s thesis is multi-turn robustness: GALILEO provides clearer protocols/metrics for **across-turn instability** and social/persuasion pressure, which DRIFT does not address.

## Where GALILEO is weaker / needs to improve

- If GALILEO discusses long-horizon agents: we may need to better cite/position **efficient long-context mechanisms** (latent token compression / memory interfaces) as enabling tech for stable multi-turn behavior.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related work: add DRIFT as “query-conditioned latent prompt compression / decoupled knowledge+reasoning” (long-context enabler).
- [ ] Writing: explicitly disambiguate **behavioral drift** vs “DRIFT” (implicit fact tokens) if both appear in the same section.
- [ ] Brainstorm: consider whether GALILEO-style multi-turn robustness metrics could be adapted to evaluate **compression robustness** (performance vs compression ratio; time-to-failure under progressively lossy context).

## Quotes / details to potentially cite

- Abstract: “DRIFT employs a lightweight knowledge model to dynamically compress document chunks into implicit fact tokens conditioned on the query… projected into the reasoning model’s embedding space, replacing raw, redundant text while maintaining inference accuracy.”
- Intro: “achieving an average 7× speedup on 256k-token documents.”
- Intro: “achieves a 32× compression ratio while improving accuracy from 20.87% to 29.22% on the LongBench v2 benchmark.”
