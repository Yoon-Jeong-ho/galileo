# Do Large Language Models Know How Much They Know?

- Year: 2024 (EMNLP main; arXiv 2025)
- Venue: EMNLP 2024 (long paper)
- Authors: Gabriele Prato, Jerry Huang, Prasanna Parthasarathi, Shagun Sodhani, Sarath Chandar
- URL: https://arxiv.org/abs/2502.19573
- BibTeX key (if we add it): prato2024knowhowmuch
- Tags: calibration, self-knowledge, uncertainty, memorization, synthetic-benchmark, recall

## One-sentence takeaway

LLMs can learn to stop recalling at the “right” point (neither omitting nor hallucinating extra items) in a synthetic multi-document recall benchmark, and this “knowledge-amount awareness” emerges with sufficient model+data scaling.

## What problem does it solve?

- Tests whether an LLM can estimate the *extent* of its knowledge about a topic (i.e., know when it is “done” retrieving all relevant memorized items), rather than only whether it knows a single fact.
- Motivated by hallucination/overconfidence/self-contradiction: if the model doesn’t know how much it knows, it may output too little or invent additional content.

## What is the core method / protocol?

- Create *synthetic* “diary entries” per fictitious individual (topic = diarist).
  - Each diarist has a random number of entries (1–8), each entry has a random number of attributes (1–8) chosen from a fixed attribute set.
- Fine-tune LLMs on:
  - all diary entry documents, and
  - 90% of Q/A pairs of the form: “Recall all of {name}’s diary entries, in order.” → concatenation of all that diarist’s entries.
  - (They note doing documents-then-QA sequentially caused catastrophic forgetting/overfitting, so they train jointly.)
- Evaluate on held-out questions: the model must output *exactly* all entries for that diarist, in order.
- Benchmark across model families/scales:
  - OPT (decoder-only), Pythia (decoder-only), Flan-T5 (encoder-decoder)
  - dataset scale: 1K–64K diarists.

## What are the key metrics?

- Exact-match accuracy on the recall task: output must match ground truth with *no errors* in:
  - the *number* of documents recalled, and
  - the content of each recalled document.
- Secondary analyses described qualitatively in intro: under-supply vs over-supply (too few vs hallucinated extra entries) as failure modes.

## What are the main results?

- With sufficient scaling (model size and/or dataset size), all tested model suites show the ability to recall the *correct amount* of information (i.e., not a random number of entries).
- Emergence differs by architecture:
  - OPT can succeed at certain sizes if the fine-tuning dataset is large enough;
  - Pythia and Flan-T5 require more scaling at comparable sizes (per the paper’s discussion around Fig. 2–3).
- The number and length of documents (within their controlled ranges) reportedly have little effect: models can memorize/retrieve both short and longer documents, and both single- and multi-document topics.

## How is this similar to GALILEO?

- Directly about *calibration / self-knowledge*: knowing “how much you know” is adjacent to knowing when to stop, uncertainty, and avoiding hallucination.
- Emphasizes *controlled synthetic benchmarks* to isolate a capability and study scaling/emergence—useful as a pattern for GALILEO evaluations.

## How is this different from GALILEO?

- Task is synthetic memorization + exhaustive recall of training documents; it does not measure real-world uncertainty or decision-making under distribution shift.
- Evaluates “completion of retrieval” in a closed world with exact-match targets, not interactive agent behavior, tool use, or long-horizon planning.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets real tasks / agents, it can connect calibration to downstream utility (safe stopping, deferral, tool calls), not just memorized document enumeration.
- GALILEO can incorporate *selective answering* and explicit abstention/deferral criteria rather than requiring verbatim exhaustive outputs.

## Where GALILEO is weaker / needs to improve

- Might lack a similarly *clean* controlled benchmark that forces the model to demonstrate “I know I’m done” behavior (vs relying on indirect uncertainty metrics).

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a “closed-world coverage” eval: given a latent set of K items associated with a topic, measure under/over-generation and the model’s ability to stop at K.
- [ ] In related work, position GALILEO vs. “knowledge awareness” work: they argue knowing a single answer differs from aggregating multiple training samples and knowing when retrieval is complete.
- [ ] If we use synthetic tasks, adopt their practice of controlling entropy/information per sentence (fixed attribute templates) to isolate length vs count effects.

## Quotes / details to potentially cite

- “we task LLMs with enumerating everything they know about a given topic—no more, no less.” (Intro)
- Setup: fine-tune on synthetic diary entries + Q/A; evaluate exact match requiring correct *quantity and content*.
- They report sequential training (docs then Q/A) led to “catastrophic forgetting” of documents and “overfitting” on Q/A, motivating joint fine-tuning.
