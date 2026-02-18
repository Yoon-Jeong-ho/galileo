# FairMT-Bench: Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs

- Year: 2024
- Venue: arXiv
- Authors: Zhiting Fan, Ruizhe Chen, Tianxiang Hu, Zuozhu Liu
- URL: https://arxiv.org/abs/2410.19317
- BibTeX key (if we add it): FairMTBench2024Fan
- Tags: fairness, bias, multi-turn, dialogue, benchmark, safety-eval

## One-sentence takeaway

FairMT-Bench introduces a multi-turn fairness benchmark (FairMT-10K / distilled FairMT-1K) showing that bias/toxicity rates often increase with dialogue turns and that models vary sharply by task type (context understanding vs bias-resistance vs instruction trade-offs).

## What problem does it solve?

- Existing fairness evaluations for LLMs are mostly single-turn, missing multi-turn phenomena like (i) bias accumulation across turns, (ii) failures from anaphora/ellipsis and scattered context, and (iii) instruction-following pressures that trade off fairness vs utility.
- Provides a more realistic benchmark setting for fairness in conversational deployments.

## What is the core method / protocol?

- Task taxonomy across three “stages” of multi-turn interaction (each with two tasks):
  - Context understanding: (1) Anaphora/Ellipsis (bias hidden via pronouns), (2) Scattered Questions (bias components distributed across turns).
  - Interaction fairness: (3) Jailbreak Tips (misleading user instructions generated to elicit biased outputs), (4) Interference from Misinformation (biased viewpoints injected earlier; final question requires using history).
  - Fairness trade-off: (5) Negative Feedback (user repeatedly negates/refutes refusals), (6) Fixed Format (structured format conditioning then final biased query).
- Dataset construction:
  - FairMT-10K multi-turn prompts (5 turns) built by integrating examples from existing human-annotated bias/toxicity sources (stereotypes + toxicity; attributes like gender/race/religion/etc.).
  - Uses GPT-4 as a “proxy user” to generate some multi-turn interactions/templates.
  - Distills FairMT-1K by selecting hardest items (highest error ratios across tested models), balanced across tasks.
- Evaluation:
  - Run the multi-turn dialogue sequentially; score only the final-turn output.
  - Primary judge: GPT-4-based rubric/judging; auxiliary detector: Llama-Guard-3; plus a human validation study to sanity-check judge alignment.

## What are the key metrics?

- “Bias ratio” / “bias rate”: proportion of dialogue instances judged biased/toxic in the final turn.
- Reported by task, bias type (stereotype vs toxicity), and by model.
- Also analyzes turn-by-turn bias ratio trends (bias accumulation with more turns).

## What are the main results?

- Multi-turn context often increases bias/toxicity vs single-turn evaluation; bias tends to rise with the number of turns, with a spike at the final turn.
- Tasks that stress contextual reference and implicit bias detection (notably Anaphora/Ellipsis, and Interference-from-Misinformation) are consistently challenging.
- Model performance is heterogeneous and task-dependent:
  - Some models do better on comprehension-oriented tasks but degrade under instruction pressure / trade-off tasks, and vice versa.
- Distilled FairMT-1K remains challenging even for a wider set of more recent models, suggesting “fairness under multi-turn pressure” is not solved.

## How is this similar to GALILEO?

- Shares the core framing that single-turn evaluation can miss important failure modes that emerge only over multi-turn interactions.
- Uses a taxonomy of multi-turn failure settings and emphasizes longitudinal/turn-based analysis rather than one-off prompts.

## How is this different from GALILEO?

- Focuses on fairness/bias/toxicity rather than agreement-seeking, belief drift, or other multi-turn robustness axes.
- Uses LLM-as-judge (GPT-4) + a safety classifier + limited human validation; if GALILEO targets more objective or mechanistic metrics, the evaluation philosophy differs.
- The benchmark is built from fairness/toxicity sources and prompt templating; it is not centered on “preference pressure” or sycophancy specifically.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses more behaviorally grounded or task-specific success metrics (beyond LLM-judge), it can reduce judge bias and improve interpretability.
- If GALILEO isolates causal factors (e.g., controlled pressure interventions), it may offer clearer mechanistic conclusions than a broad benchmark suite.

## Where GALILEO is weaker / needs to improve

- FairMT-Bench’s explicit decomposition into (context understanding vs interaction interference vs trade-offs) is a useful template; if GALILEO’s related-work taxonomy is less explicit, adopting a similar structure could sharpen the narrative.
- FairMT-Bench reports bias trends by turn count and by social attribute; GALILEO could add analogous stratified analyses (by “pressure type”, “turn index”, “topic class”, etc.).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a related-work paragraph positioning multi-turn robustness as including fairness/bias accumulation, citing FairMT-Bench as evidence that multi-turn context can systematically worsen safety-aligned behavior.
- [ ] Consider a “reference resolution / scattered context” stress test (pronouns, ellipsis, split evidence across turns) as a general multi-turn robustness probe.
- [ ] If using LLM-as-judge, explicitly include a small human-check protocol and/or an auxiliary classifier to triangulate.

## Quotes / details to potentially cite

- “Existing fairness benchmarks mainly focus on single-turn dialogues, while multi-turn scenarios … pose greater challenges due to conversational complexity and risk for bias accumulation.”
- Task taxonomy stages: “context understanding, interaction fairness, and fairness trade-offs.”
- Empirical claim: bias ratios generally increase with number of turns (bias accumulation), with certain tasks (e.g., anaphora/ellipsis) especially challenging.
