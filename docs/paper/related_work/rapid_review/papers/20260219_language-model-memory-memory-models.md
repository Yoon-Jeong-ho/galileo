# Language Model Memory and Memory Models for Language

- Year: 2026
- Venue: arXiv
- Authors: Benjamin Badger
- URL: https://arxiv.org/abs/2602.13466
- BibTeX key (if we add it): Badger2026LanguageModelMemory
- Tags: memory embeddings, invertibility, autoencoders, compression, encoder-decoder, objectives, copy-task

## One-sentence takeaway

Next-token-prediction embeddings are generally information-poor (hard to invert/reconstruct inputs), while autoencoder-style objectives produce information-rich “memories”; combining causal + information-retention objectives (and/or frozen high-fidelity encoders + curriculum) yields memory models that can both predict next tokens and support more faithful memory retrieval.

## What problem does it solve?

- Clarifies *how much input information* is retained in a language model’s “memory” (often operationalized as the last-layer embedding of the last token), and why single-embedding “memory models” often underperform full-context Transformers.
- Motivates training/architectural changes so that replacing token sequences with compressed embeddings does not destroy the ability to access arbitrary details from the original input.

## What is the core method / protocol?

- Define “memory” as the last hidden-layer embedding (last token) for an input sequence.
- Measure information retention by *inversion*: freeze an encoder (e.g., causal LM / masked model / retrieval model / autoencoder), train a decoder to reconstruct the original token sequence from the embedding (using an “unrolling” projection from the single embedding to a sequence of decoder inputs).
- Report retention with two tokenizer-size-normalized metrics:
  - Entropy-ratio-style normalized cross-entropy (fraction of information retained vs a uniform baseline).
  - Token identification accuracy (Hamming-style, % tokens exactly recovered).
- Propose a parallelizable “memory model” architecture: chunk the sequence, encode chunks to embeddings in parallel, then have a decoder consume embeddings (+ possibly remaining tokens) to do next-token prediction.
- Show that *causal-only* training yields poor memory embeddings; improve with:
  - Combined objective: next-token prediction + copy-style retention objective.
  - Curriculum for frozen encoders: train decoder to use memories first (blank-copy), then add standard causal next-token prediction.

## What are the key metrics?

- Input reconstruction quality from embeddings:
  - Normalized cross-entropy / “entropy ratio” retention score.
  - Token reconstruction accuracy (% correct tokens).
- Copy / blank-copy accuracies for memory models (tests whether decoder can use memory embeddings to reproduce withheld tokens).
- Secondary: benchmark accuracy changes when adapting pretrained decoders (e.g., Llama) to memory embeddings.

## What are the main results?

- For typical causal / retrieval / masked LMs, single “memory” embeddings tend to contain relatively little recoverable information about the full input sequence; increasing model/data/compute helps only modestly, and longer/more diverse contexts are harder to invert.
- Autoencoders trained to regenerate inputs can produce near-lossless memories (high-fidelity reconstruction), and this behavior is non-trivial (does not generalize to uniform-random token sequences; shows some OOD generalization).
- Causal-trained memory-model encoders (chunk encoders) remain information-poor and fail harder “blank copy” tests.
- Memory models can be trained to both:
  - Predict next tokens efficiently (benefits: lower KV cache / time-to-first-token when using embeddings instead of full token histories), and
  - Form/use information-rich memories, but this typically requires combined objectives or training curricula (especially when decoders can otherwise ignore memory embeddings and take an easier local minimum).

## How is this similar to GALILEO?

- Directly engages with the theme that compressed internal state (embeddings / summaries / memories) can substitute for full token histories for efficiency—*but only if* the representation supports retrieval of arbitrary details.
- Emphasizes evaluation protocols for “memory quality” beyond downstream task accuracy, via reconstruction/information retention and copy-like diagnostics.

## How is this different from GALILEO?

- Focuses on *single-embedding* (or chunk-embedding) memory formation and information-theoretic/invertibility framing, rather than (presumably) GALILEO’s broader agentic / tool / multi-component pipeline behavior.
- Uses inversion/reconstruction as a primary lens; GALILEO may care more about functional recall/reasoning utility under task distributions than strict reconstructability.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets task-relevant recall rather than full input reconstruction, it may achieve better compute/utility tradeoffs than a near-lossless memory requirement.
- If GALILEO uses external memory (retrieval, structured stores), it may avoid overloading a single embedding with arbitrary information.

## Where GALILEO is weaker / needs to improve

- If GALILEO relies heavily on compressed representations (summaries/latent memories) without an auxiliary retention objective, it may inherit the “information-poor embedding” failure mode highlighted here.
- If evaluation focuses on next-token/performance metrics only, it may miss latent memory degradation that harms rare-detail queries.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a *memory fidelity* diagnostic suite: reconstruct / answer arbitrary-detail queries about earlier context given only compressed state; track retention vs context length and domain diversity.
- [ ] Consider *combined objectives* for any learned compressor/memory module (task loss + retention/copy/autoencoding-style auxiliary) to prevent information collapse.
- [ ] If using frozen encoders or pretrained decoders, try a *curriculum*: train decoder to rely on memory-only inputs before mixing in token-level signals.
- [ ] In writing: cite the argument that next-token prediction is many-to-one / “non-invertible” and therefore poorly suited to induce high-fidelity memories without extra objectives.

## Quotes / details to potentially cite

- Claim/observation: “language model embeddings typically contain relatively little input information regardless of data and compute scale during training,” contrasted with near-perfect memory formation from autoencoders trained for regeneration.
- Rationale: next-token prediction is many-to-one and “poorly suited for accurate memory formation” because the objective is effectively non-invertible.
- Complexity note: for chunked memory models, encoder/decoder compute can scale better than full attention (derives an ideal chunk count scaling like s \u2248 n^(2/3) under their assumptions), enabling inference savings (KV cache/time-to-first-token).
