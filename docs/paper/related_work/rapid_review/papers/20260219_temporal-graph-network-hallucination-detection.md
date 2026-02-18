# Temporal Graph Network: Hallucination Detection in Multi-Turn Conversation

- Year: 2026
- Venue: arXiv
- Authors: Vidhi Rathore; Sambu Aneesh; Himanshu Singh
- URL: https://arxiv.org/abs/2601.03051
- BibTeX key (if we add it): rathore2026temporalgraph
- Tags: hallucination-detection, multi-turn, graph-neural-net, dialogue-consistency

## One-sentence takeaway

Represent a multi-turn dialogue as a temporal+entity-linked graph over turns, run GNN message passing + attention pooling, and classify dialogue-level hallucination type (slightly) better than a prior graph baseline on DiaHalu.

## What problem does it solve?

- Detect *dialogue-level* hallucinations/inconsistencies in multi-turn conversations, where contradictions or unsupported claims may only become apparent after context evolves.
- Avoid brittle/heavy pipelines (e.g., claim extraction + retrieval + verification) by using a lightweight, self-contained representation of conversational structure.

## What is the core method / protocol?

- Build a graph where each node is a dialogue turn (utterance) represented by a sentence embedding (sentence-transformer; they mention All MiniLM L6 v2).
- Add two edge types:
  - Temporal (directed) edges between consecutive turns.
  - Shared-entity (undirected) edges between turns that mention the same named entities.
- Run message passing with a GNN to produce context-aware turn embeddings.
- Aggregate all node embeddings with attention pooling into one dialogue representation.
- Classify into one of six hallucination categories from DiaHalu.

## What are the key metrics?

- Multiclass: accuracy, weighted F1.
- Binary (hallucination vs not): accuracy, F1.
- They report averages over 25 runs (and std devs).

## What are the main results?

- Evaluated on DiaHalu (80/20 split).
- Their TGN variants outperform the chosen baseline (GCA; an RGCN + sentence encoder) on **binary accuracy**, with **comparable F1**; and TGN additionally predicts hallucination type.
- Ablations suggest shared-entity edges are particularly helpful:
  - TGN[ET] (temporal+entity) achieves highest multiclass accuracy.
  - TGN[E] (entity edges only) attains best multiclass weighted F1 and binary accuracy.
- Under-represented classes (e.g., incoherence/irrelevance/overreliance) remain challenging.

## How is this similar to GALILEO?

- Targets *multi-turn* failure modes where errors accumulate over turns (context drift / inconsistency).
- Emphasizes structured evaluation beyond single-turn checks; uses a benchmark with multi-turn annotations.
- Uses model-internal signals (representations + pooling/attention) to localize which turns matter (useful for analysis-style reporting).

## How is this different from GALILEO?

- Focus is *hallucination detection/classification* in dialogues, not robustness to social pressure, persuasion, or belief/stance drift under adversarial prompting.
- Supervised classification on a labeled dataset (DiaHalu) rather than stress-testing a generation protocol under interactive pressure.
- Uses a relatively conventional architecture (sentence embeddings + GNN) rather than a multi-round robustness protocol for LLM agents.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s goal is robustness/stability under pressure (not just detecting hallucinations), it can claim broader coverage of multi-turn *behavioral* failure modes (sycophancy/persuasion/belief drift), whereas this paper is narrowly about hallucination categories.
- GALILEO likely avoids dependence on a particular NER/extraction pipeline (this work’s shared-entity edges require entity extraction quality; they mention using an external model for entity extraction).

## Where GALILEO is weaker / needs to improve

- GALILEO may need a clearer “structured conversation representation” story for multi-turn analysis; this paper provides a simple, explainable graph construction that readers can immediately visualize.
- GALILEO could benefit from adding a hallucination/contradiction tracking component (or at least citing this line of work) when arguing about multi-turn failure accumulation.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite graph-based dialogue-structure approaches as a lightweight alternative to retrieval/verification pipelines for multi-turn inconsistency detection.
- [ ] Consider an analysis figure: represent GALILEO dialogues as graphs (temporal + entity/topic edges) and measure where/when failures emerge (turn position, entity re-mention, contradiction points).
- [ ] If GALILEO has multi-turn labels (or can derive them), try a diagnostic: do entity-linked edges predict higher risk of drift/flip in later turns?

## Quotes / details to potentially cite

- “By representing the entire conversation as a temporal graph, we present a novel graph-based method for detecting dialogue-level hallucinations.”
- Edges: “shared-entity edges … connect turns that refer to the same entities; … temporal edges … connect contiguous turns.”
- Dataset: “we use the DiaHalu dataset … a benchmark designed for dialogue-level hallucination detection … multi-turn conversations annotated with the types of hallucinations.”
- They report: “All reported results are averaged over 25 runs to ensure stability of results.”
