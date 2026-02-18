# Talk Less, Verify More: Improving LLM Assistants with Semantic Checks and Execution Feedback

- Year: 2026
- Venue: WITS 2025 (Workshop on Information Technologies and Systems)
- Authors: Yan Sun et al.
- URL: https://arxiv.org/abs/2601.00224
- BibTeX key (if we add it): Sun2026TalkLessVerifyMore
- Tags: verification, execution-feedback, code-generation, semantic-alignment, generator-discriminator

## One-sentence takeaway

Add an explicit verifier (reverse-translation semantic check) and execution-feedback loop around LLM code generation to reduce wrong-but-plausible answers and speed up task completion on text-to-SQL and math benchmarks.

## What problem does it solve?

- In enterprise / conversational business analytics settings, LLM assistants often output code/results that *look* reasonable but are semantically misaligned with the user’s intent, and users must manually verify and iterate.
- The paper targets reducing semantic errors and improving executability by moving verification inside the system loop.

## What is the core method / protocol?

- Two verification mechanisms embedded in a generator–discriminator style loop:
  - **Q\***: “reverse translation” from generated code back into a natural-language description, then **semantic matching** between that description and the user’s original intent (a discriminator-style check).
  - **Feedback+**: run / execute the generated artifact and use **execution feedback** (errors, failures) to guide refinement.
- The overall design is positioned as shifting validation responsibility from the end user to the assistant/system.

## What are the key metrics?

- Error rate / correctness on benchmark tasks.
- Task completion time (user time / iteration time) as a usability-oriented metric.

## What are the main results?

- On **Spider**, **BIRD**, and **GSM8K**, both Q\* and Feedback+ reduce error rates and reduce task completion time versus a baseline generation-only workflow.
- Reverse translation quality/cost is highlighted as a bottleneck in the approach.

## How is this similar to GALILEO?

- Shares the central motivation of **reliability**: reducing silent failures and improving trustworthiness of assistant outputs.
- Uses an explicit **check-and-revise loop**, which is conceptually aligned with agentic verification / self-correction patterns.

## How is this different from GALILEO?

- Focuses on **semantic alignment between intent and code** (via reverse translation + semantic matching) and **execution error feedback**, rather than GALILEO’s core contributions (as framed in our paper) around dataset/task design and the specific evaluation protocol for robust agent behavior.
- Evaluates on classic text-to-SQL and math QA benchmarks rather than GALILEO’s targeted setting.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO emphasizes controlled protocols/benchmarks for agent evaluation, it can provide clearer attribution and more comprehensive failure taxonomy than an application-driven CBA pipeline.

## Where GALILEO is weaker / needs to improve

- This paper suggests adding “enterprise-style” guardrails (semantic checks + execution feedback) can improve practical usefulness; if GALILEO agents currently lack systematic semantic verification, this is a gap.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite as an example of **verification layers** for LLM assistants (reverse translation + execution feedback) and contrast with GALILEO’s focus.
- [ ] Consider a small ablation or discussion: “semantic intent verification” vs “environment/execution verification” and where GALILEO fits.
- [ ] Add a short paragraph noting reverse-translation as a potential bottleneck (cost/quality) in verifier-based pipelines.

## Quotes / details to potentially cite

- “Q\* … performs reverse translation and semantic matching between code and user intent.”
- “Feedback+ … incorporates execution feedback to guide code refinement.”
- Evaluated on “Spider, Bird, and GSM8K” with reductions in “error rates and task completion time.”
