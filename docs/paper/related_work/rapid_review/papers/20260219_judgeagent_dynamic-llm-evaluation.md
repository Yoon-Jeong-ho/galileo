# JudgeAgent: Beyond Static Benchmarks for Knowledge-Driven and Dynamic LLM Evaluation

- Year: 2026
- Venue: arXiv
- Authors: Zhichao Shi; Xuhui Jiang; Chengjin Xu; Cangli Yao; Shengjia Ma; Yinghan Shen; Zixuan Li; Jian Guo; Yuanzhuo Wang
- URL: https://arxiv.org/abs/2509.02097
- BibTeX key (if we add it): Shi2026JudgeAgent
- Tags: dynamic-eval, adaptive-testing, context-graph, llm-as-judge, multi-turn

## One-sentence takeaway

JudgeAgent is a dynamic, knowledge-structured (context-graph) evaluator that adaptively interviews an LLM with multi-turn, difficulty-controlled questions to broaden knowledge coverage and reduce static-benchmark saturation/contamination.

## What problem does it solve?

- Static benchmarks cover limited slices of knowledge and quickly saturate; they also risk train/test contamination.
- Existing “dynamic” evaluators still rely heavily on an evaluator LLM’s ad-hoc generations, yielding limited/unstable knowledge coverage and non-adaptive (mismatched) difficulty.

## What is the core method / protocol?

- Three-stage evaluation pipeline:
  - **(1) Benchmark grading:** run the target on an initial static benchmark in small batches; compute a capability estimate and map it to an Easy/Med/Hard difficulty via a **linear difficulty control** mechanism.
  - **(2) Interactive extension (multi-turn interview):** iteratively generate follow-up questions that both (a) extend knowledge beyond the seed question and (b) adapt difficulty based on the evolving capability estimate.
    - **Knowledge-driven retrieval via a context graph:** build a graph over a benchmark knowledge base (entities/chunks as nodes; links if co-occur in texts). For a seed question, extract entities, find similar entities on the graph, and sample multi-hop paths; concatenate retrieved chunks as background to condition question generation.
    - **Difficulty-adaptive question generation:** prompt the evaluator to generate questions under explicit difficulty-specific requirements (easy retrieval/cloze → medium conceptual/inference → hard multi-step reasoning).
  - **(3) Evaluation feedback:** produce an evaluation report over the Q&A history that highlights knowledge deficiencies and gives actionable suggestions.

## What are the key metrics?

- Primary validation is **indirect**: use JudgeAgent’s evaluation report as “suggestions” to re-prompt the target to answer the same questions again, and measure **accuracy improvement** pre vs. post intervention.
- Additional diagnostics discussed conceptually: breadth/depth of covered knowledge (via context-graph traversal) and alignment of question difficulty to target capability (via adaptive control).

## What are the main results?

- The paper reports that JudgeAgent enables more comprehensive evaluation and produces feedback that can drive effective model iterations (measured via post-suggestion accuracy gains in their setup).
- Experimental setup (from the HTML): evaluator core model is GPT-4.1 (API), with interactive extension capped to a small number of rounds; initial benchmarks include MedQA, MultiHop-RAG, and QuALITY.

## How is this similar to GALILEO?

- Shares the theme of **interactive / multi-turn evaluation** rather than single-shot static scoring.
- Uses structured context (a graph over knowledge) to make evaluation more systematic than pure prompt sampling.

## How is this different from GALILEO?

- Focus is on **knowledge evaluation** (dynamic interviewing + knowledge-graph guided question generation), not on agent/task performance per se.
- Uses an explicit **difficulty-control** mechanism (easy/medium/hard thresholds) and frames evaluation as an “interview” protocol.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO evaluates agents in realistic environments/tools, that can be a more direct measure of end-to-end capabilities than knowledge-only interview-style tests.
- If GALILEO has clearer, standardized outcome metrics (task success, efficiency, safety), it may avoid the “indirect validation via re-prompting” that can be gamed.

## Where GALILEO is weaker / needs to improve

- Consider adding **difficulty-adaptive** evaluation to avoid ceiling/floor effects across model versions.
- Consider systematic **coverage expansion** from seed items (graph-based or taxonomy-based) rather than relying on evaluator LLM randomness.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite JudgeAgent as a representative of **knowledge-driven dynamic evaluation** (context-graph + adaptive interview).
- [ ] Consider a GALILEO ablation/appendix idea: difficulty-adaptive question/task selection (e.g., curriculum during eval) to reduce mismatch across target models.
- [ ] If GALILEO synthesizes eval items, consider “coverage accounting” (breadth/depth) analogous to their context-graph traversal narrative.

## Quotes / details to potentially cite

- “Current evaluation methods… rely on static benchmarks… limited knowledge coverage and fixed difficulties…” (Abstract)
- JudgeAgent “leverages LLM agents equipped with context graphs… [and] a difficulty-adaptive and multi-turn interview mechanism.” (Abstract)
- Method components: “Benchmark Grading… Interactive Extension… Evaluation Feedback.” (Methodology)
