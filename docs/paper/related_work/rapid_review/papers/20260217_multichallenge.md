# MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs

- Year: 2025
- Venue: arXiv (preprint)
- Authors: Ved Sirdeshmukh, Kaustubh Deshpande, Johannes Mols, Lifeng Jin, Ed-Yeremai Cardona, Dean Lee, Jeremy Kritz, Willow Primack, Summer Yue, Chen Xing
- URL: https://arxiv.org/abs/2501.17399
- BibTeX key (if we add it): sirdeshmukh2025multichallenge
- Tags: multi-turn, benchmark, instruction-retention, memory, versioned-editing, self-coherence, sycophancy-adjacent, llm-judge

## One-sentence takeaway

MultiChallenge is a realistic 4-task, up-to-10-turn benchmark for multi-turn conversations, showing frontier LLMs score <50% and proposing an **instance-level binary rubric** that enables more reliable LLM-as-judge evaluation.

## What problem does it solve?

- Existing multi-turn chat evals (e.g., MT-Bench-style) are increasingly saturated and/or focus on narrow instruction-following artifacts rather than realistic mixed requirements.
- Need a benchmark where success requires **instruction retention + context allocation + in-context reasoning** across a conversation.
- Need an automatic evaluation approach that aligns better with humans than “judge the whole dialogue” prompting.

## What is the core method / protocol?

- Dataset of test examples, each:
  - a multi-turn conversation history (max ~10 turns) ending in a final user request
  - model must respond to the final user request **given the history**
- Four challenge categories:
  - **Instruction retention**: follow first-turn instructions throughout conversation (no later conflicting instructions)
  - **Inference memory (user info)**: recall + connect earlier user details that are *implicitly* needed for the final request
  - **Reliable versioned editing**: iterative edits with references to prior versions; resolve version references; copy/edit without hallucinating
  - **Self-coherence**: remain coherent with the assistant’s earlier statements; avoid collapsing into “agreeing with the user” when the user contradicts prior assistant content (sycophancy-like)
- Construction: hybrid pipeline with multi-agent synthetic conversation generation + human review/editing.
- Automatic evaluation: **instance-level rubrics**
  - for each example, a human writes a *binary* yes/no rubric question that (by design) can be answered using only the final model response
  - use an LLM judge to answer the rubric question.

## What are the key metrics?

- Primary: **accuracy / pass rate** on binary rubric (per-example success/failure), aggregated overall and (presumably) by category.
- Judge reliability: alignment of LLM judge vs experienced human raters.
  - Reported: instance-level rubric judging reaches **93% alignment** vs **36%** when prompting judges with raw conversation context.

## What are the main results?

- All evaluated frontier models achieve **<50%** average accuracy on MultiChallenge.
- Best reported model: **Claude 3.5 Sonnet (June 2024)** at **41.4%** average accuracy.
- The instance-level rubric design materially improves LLM-as-judge agreement with humans (93% vs 36%).

## How is this similar to GALILEO?

- Targets **multi-turn robustness** failures that emerge only with conversational history.
- Includes an explicit **avoid sycophancy / maintain self-consistency under user pushback** axis (Self-coherence), adjacent to social-pressure drift.
- Useful precedent for “don’t trust naive LLM-as-judge on hard multi-turn tasks”; you need careful rubric/controls.

## How is this different from GALILEO?

- Not primarily about **social pressure / persuasion dynamics** or belief drift vs evidence-driven revision.
- No explicit time-to-failure / survival framing; each item is a *single final-turn decision* conditioned on history.
- “Self-coherence” is about consistency with prior assistant turns, not necessarily about resisting *normative pressure* or tracking recovery after being misled.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes paired neutral-vs-pressure controls, explicit drift-vs-revision separations, and flip/recovery trajectory metrics, it is more diagnostic for **why** models drift and **whether** they can recover.

## Where GALILEO is weaker / needs to improve

- MultiChallenge demonstrates a practical pattern for **auditable automatic evaluation** (binary rubrics answerable from final output) that may be more reliable than “judge full dialogue”.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adopting “instance-level binary rubric” style questions for at least a subset of GALILEO tasks to improve judge reliability and reduce judge dependence.
- [ ] Add a brief related-work note: naive LLM-as-judge on multi-turn histories can be unreliable; justify any rubric/labeling choices we make.
- [ ] If we have a self-coherence / anti-sycophancy slice, cite MultiChallenge as a realism-motivated benchmark where that axis is one of four core difficulties.

## Quotes / details to potentially cite

- “Each test example … is a maximum 10-turn conversation history … ending with a final user turn …”
- “All frontier models have less than 50% accuracy … top-performing Claude 3.5 Sonnet … 41.4% …”
- Instance-level rubrics yield “93% alignment” vs “36%” for direct-judge-with-raw-context prompting.
