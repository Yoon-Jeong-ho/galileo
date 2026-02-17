# MultiVerse: A Multi-Turn Conversation Benchmark for Evaluating Large Vision and Language Models

- Year: 2025
- Venue: arXiv
- Authors: Young-Jun Lee, Byung-Kwan Lee, Jianshu Zhang, Yechan Hwang, Byungsoo Ko, Han-Gyu Kim, Dongyu Yao, Xuankun Rong, Eojin Joo, Seung-Ho Han, Bowon Ko, Ho-Jin Choi
- URL: https://arxiv.org/abs/2510.16641
- BibTeX key (if we add it): multiverse2025
- Tags: multi-turn, benchmark, VLM, conversation, evaluation, checklist-eval

## One-sentence takeaway

MultiVerse is a 647-dialogue benchmark (avg ~4 turns) that stress-tests VLMs on multi-turn interaction goals and uses a GPT-4o checklist-style rubric over 37 aspects, finding even top models only ~50% succeed on complex dialogues.

## What problem does it solve?

- Single-turn VLM benchmarks overestimate real-world readiness; many applications require multi-turn clarification, goal pursuit, and consistency across turns.
- Prior multi-turn VLM sets (per the paper) cover only subsets of conversational scenarios; MultiVerse aims for broader coverage and explicit interaction goals.

## What is the core method / protocol?

- Construct a multi-turn benchmark by deriving dialogues from 12 existing VLM evaluation benchmarks.
- Scale/shape: 647 dialogues, averaging ~4 turns.
- Defines 484 tasks and 484 interaction goals.
- Evaluation: a checklist-based automated evaluation using GPT-4o as judge, scoring performance across 37 “aspects” (e.g., perceptual accuracy, linguistic clarity, factual correctness).
- Reports results over 18 VLMs; also studies the effect of providing full dialogue context (context availability) on performance.

## What are the key metrics?

- Checklist-based success / aspect scores (37 aspects), with an overall “success rate” reported in the abstract.
- Comparisons across models and ablations on context provision (full dialogue context vs limited context).

## What are the main results?

- Even the strongest evaluated model (GPT-4o, per abstract) achieves only ~50% success on “complex” multi-turn conversations.
- Providing full dialogue context helps smaller/weaker models substantially, suggesting strong in-context learning dependence in multi-turn settings.

## How is this similar to GALILEO?

- Shared emphasis on *multi-turn* evaluation rather than single-turn scores.
- Highlights that evaluation protocol details (e.g., what context is available) can dominate conclusions—relevant for designing robust, reproducible multi-turn evaluation in GALILEO.

## How is this different from GALILEO?

- Focuses on VLMs and multi-turn *task/dialogue goal* completion, not specifically persuasion/sycophancy/social-pressure dynamics.
- Uses an LLM-judge (GPT-4o) checklist rubric; GALILEO may need more judge-robust metrics or human-verified components depending on claims.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets sycophancy/persuasion robustness, it may offer a more *behaviorally-specific* protocol (pressure, drift, recovery) than general multi-turn goal benchmarks.

## Where GALILEO is weaker / needs to improve

- MultiVerse suggests a way to structure *interaction goals* and evaluate across many “aspects”; if GALILEO lacks a clear goal taxonomy/aspect rubric, it could feel narrower.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding an “interaction goal” field to multi-turn episodes (what the user is trying to get the assistant to do) to improve stratified analysis.
- [ ] If using LLM judges, consider a checklist/rubric approach with multiple aspects + report judge sensitivity (model/version, prompt) to avoid brittle conclusions.
- [ ] Add an ablation: full-history context vs truncated/partial context to quantify how much multi-turn behavior depends on long-context access.

## Quotes / details to potentially cite

- Benchmark scale and construction: “647 dialogues - each averaging four turns - derived from … 12 … VLM evaluation benchmarks.”
- Evaluation approach: “checklist-based evaluation … using GPT-4o … measuring … 37 key aspects …”
- Headline result: “even the strongest models (e.g., GPT-4o) achieve only a 50% success rate …”
