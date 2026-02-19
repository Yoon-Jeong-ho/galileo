# Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs

- Year: 2025
- Venue: arXiv
- Authors: Mohammad Tavakoli; Alireza Salemi; Carrie Ye; Mohamed Abdalla; Hamed Zamani; J. Ross Mitchell
- URL: https://arxiv.org/html/2510.27246v1
- BibTeX key (if we add it): Tavakoli2025BeyondMillionTokens
- Tags: long-context, long-term-memory, benchmark, enhancement

## One-sentence takeaway

BEAM is a synthetic-but-human-validated long-conversation benchmark (up to ~10M tokens) for diverse “memory abilities”, and LIGHT is a cognition-inspired recipe (episodic retrieval + working memory + scratchpad) that modestly but consistently improves LLM performance as dialogue length grows.

## What problem does it solve?

- Existing long-context / long-term-memory benchmarks often lack coherent long-range narrative structure (stitched short dialogs), cover narrow domains, and mostly test simple recall.
- There is also limited guidance on *how* to augment LLMs to better answer questions requiring long-range conversational memory.

## What is the core method / protocol?

- **BEAM benchmark construction** (automatic generation + validation):
  - Generate long, topically-diverse, coherent user–assistant conversations via a hierarchical “conversation plan” that is recursively refined into sub-plans and then into chronologically ordered turns.
  - Generate probing questions targeting **10 memory dimensions** (paper framing) with an emphasis on multi-hop / non-trivial reasoning.
  - Human validation of questions for quality.
  - Scale: **100 conversations**, lengths reported in the paper as **~100K to ~10M tokens**, and **2,000** probing questions.
- **LIGHT framework** (to improve performance on BEAM-style probing):
  - **Episodic memory**: an index of the full conversation, used for retrieval of relevant past content.
  - **Working memory**: recent turns kept directly in-context.
  - **Scratchpad**: a running set of salient facts extracted/updated after each turn, intended to support later recall and consistency.
  - At inference time, the model answers using a combination of retrieved episodic content + working window + accumulated scratchpad.

## What are the key metrics?

- Performance on probing questions over long conversations (reported as an aggregate question-answering score/accuracy across BEAM); paper emphasizes degradation as dialogue length increases.
- Deltas vs baselines (with/without retrieval augmentation; across different backbone LLMs).

## What are the main results?

- Even models with **~1M-token context windows** (with or without retrieval augmentation) still struggle as conversations get longer (performance drops with length).
- LIGHT provides consistent gains over the strongest baselines, reported as **~3.5% to 12.69% average improvement** depending on the backbone LLM.
- Ablations indicate each of the three memory components contributes to overall gains.

## How is this similar to GALILEO?

- Shares the core premise that **long-horizon interaction requires explicit memory mechanisms** beyond naïvely extending context.
- Uses a **structured memory decomposition** (short-term vs long-term + distilled state) that resembles common agent architectures GALILEO might adopt.
- Emphasizes *evaluation* that stresses long-range consistency and non-trivial memory use rather than only short-context QA.

## How is this different from GALILEO?

- BEAM is primarily a **benchmark + synthetic data generation pipeline** for long conversations, whereas GALILEO likely targets a broader system/agent capability set (and may emphasize real tasks or tool use).
- LIGHT is presented as a **framework/recipe**; it is not clearly positioned as an end-to-end learned memory module trained jointly with the model.
- The work is centered on **conversational probing questions**; depending on GALILEO’s scope, it may need task-oriented, tool-augmented, or environment-interactive evaluations.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses real-world tasks / tool traces, it may avoid some of the “synthetic conversation” critiques while preserving long-horizon demands.
- If GALILEO has a clearer operationalization of memory updates (when to write, what to write, how to validate), it can be cleaner than generic “scratchpad facts” extraction.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a **length-scaling evaluation** (e.g., controlled sweeps from 100K → 1M+ tokens) it may be weaker than BEAM on stress-testing memory vs length.
- If GALILEO doesn’t explicitly separate **episodic retrieval vs working window vs distilled state**, it may underperform or be harder to analyze.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a brief related-work paragraph contrasting “bigger context windows” vs “explicit memory systems”, citing this paper’s finding that 1M-token windows still degrade with length.
- [ ] Consider adopting LIGHT-like **three-way memory separation** as a baseline configuration in GALILEO experiments.
- [ ] If GALILEO has its own benchmark, add a **length scaling curve** to show robustness as context grows.

## Quotes / details to potentially cite

- BEAM scale: “**100** conversations” and “**2,000** validated questions”; conversation lengths “**100K to 10M tokens**”.
- High-level claim: even “**1M token context windows** (with and without retrieval augmentation) struggle as dialogues lengthen.”
- LIGHT improvement: “average improvement of **3.5%–12.69%** over the strongest baselines (depending on backbone LLM).”
