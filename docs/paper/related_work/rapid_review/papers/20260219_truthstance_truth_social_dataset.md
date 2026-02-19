# TruthStance: An Annotated Dataset of Conversations on Truth Social

- Year: 2026
- Venue: arXiv
- Authors: Fathima Ameen; Danielle Brown; Manusha Malgareddy; Amanul Haque
- URL: https://arxiv.org/abs/2602.14406
- BibTeX key (if we add it): ameen2026truthstance
- Tags: dataset, stance-detection, argument-mining, conversation-trees, alt-tech, truth-social, llm-annotation

## One-sentence takeaway

TruthStance releases large-scale Truth Social conversation trees plus a smaller human-labeled benchmark for argument presence and parent-relative stance, and uses LLM prompting to scale labels over the full corpus.

## What problem does it solve?

- Existing conversational stance/argument datasets largely focus on mainstream platforms (Twitter/Reddit), while Truth Social (alt-tech) lacks publicly available conversation-thread corpora with reply structure.
- Enables studying stance dynamics in deep reply trees (where stance is hard to infer without intermediate context).

## What is the core method / protocol?

- Data collection: extend an existing Truth Social post-level dataset by scraping comments and preserving full reply-tree (thread) structure (2023–2025).
- Two annotation tasks:
  - Argument mining on root posts: whether the post contains an argument (argument presence).
  - Claim-based stance detection on comments: stance of a reply relative to its immediate parent in the conversation (enables tree traversal to infer stance vs. root by composition).
- Benchmarking/labeling strategy:
  - Create a human-annotated benchmark (1,500 instances) for both tasks, report inter-annotator agreement.
  - Evaluate LLM prompting strategies on the benchmark; use best-performing prompt/config to produce large-scale LLM-generated labels over the broader dataset (posts + a subset of comments).

## What are the key metrics?

- Task performance on the human-labeled benchmark for:
  - Argument presence classification.
  - Parent-relative stance classification.
- Inter-annotator agreement for the benchmark labels.
(Exact scores not captured in this rapid pass; see paper for numbers.)

## What are the main results?

- Dataset release:
  - 24,352 conversation threads (root posts) with 523,360 comments and reply-tree structure.
- Labels released:
  - Human labels: 1,500 instances spanning the two tasks.
  - LLM labels (using best prompt):
    - 24,352 posts labeled for argument presence.
    - 107,873 comments labeled for stance-to-parent.
- Provides initial analyses of stance/argument patterns across conversation depth, topics, and users (qualitative + quantitative; details in paper).

## How is this similar to GALILEO?

- Treats discourse understanding as structured prediction over conversations (reply trees), rather than isolated posts.
- Uses LLMs as scalable annotators/evaluators, with a smaller human-validated benchmark to select prompting strategies.

## How is this different from GALILEO?

- Primary contribution is a domain-specific dataset (Truth Social) + annotation pipeline, not a new modeling framework.
- Stance is framed locally (reply-to-parent) and then composed over the tree; GALILEO may instead target different structure/targets (e.g., document-level goals, tool-augmented reasoning, or alternative interaction schemas).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO emphasizes generalization/robustness across domains, it can position this work as a single-platform case study.
- If GALILEO provides explicit mechanisms for global consistency across a tree (beyond local parent stance), that is a clear differentiator.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a comparable large-scale conversational dataset setting (deep trees, stance propagation), this paper is a reminder to include conversation-structure evaluations or at least discuss them.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite TruthStance as (i) an alt-tech conversational dataset and (ii) an example of parent-relative stance labeling that can be composed along a reply tree.
- [ ] Consider a small discussion paragraph: local (edge-level) labeling vs. global (root-level) stance inference; how GALILEO would enforce consistency across paths.
- [ ] If GALILEO uses LLM-based labeling/evaluation, mention their benchmark-first-then-scale protocol as a pragmatic pattern (and note limitations).

## Quotes / details to potentially cite

- “TruthStance, a large-scale dataset of Truth Social conversation threads spanning 2023–2025 … 24,352 … posts and 523,360 comments with reply-tree structure preserved.” (Abstract)
- Human benchmark: “1,500 instances across argument mining and claim-based stance detection … evaluate large language model (LLM) prompting strategies.” (Abstract)
- Scaled labels: “additional LLM-generated labels for 24,352 posts (argument presence) and 107,873 comments (stance to parent).” (Abstract)
