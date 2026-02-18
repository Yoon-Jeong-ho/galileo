# HumanLLM: Benchmarking and Improving LLM Anthropomorphism via Human Cognitive Patterns

- Year: 2026
- Venue: arXiv
- Authors: Xintao Wang, Jian Yang, Weiyuan Li, Rui Xie, Jen-tse Huang, Jun Gao, Shuai Huang, Yueping Kang, Liyuan Gou, Hongwei Feng, Yanghua Xiao
- URL: https://arxiv.org/abs/2601.10198
- BibTeX key (if we add it): Wang2026HumanLLM
- Tags: anthropomorphism, role-play agents, personality drift, multi-pattern dynamics, evaluation, checklists

## One-sentence takeaway

HumanLLM builds a psychology-pattern dataset + checklist-based evaluation to train and test whether LLM agents can *consistently* express interacting cognitive/personality patterns across multi-turn scenarios (and argues that single holistic “human-likeness” scores confound fidelity with social desirability).

## What problem does it solve?

- RPLA / persona simulation methods often treat traits as isolated labels, leading to inconsistency (“personality illusion” / drift) and poor modeling of *interactions* among cognitive/social patterns.
- Existing evaluations can be overly holistic, blending “accurate simulation” with “nice/acceptable behavior,” and missing failure modes in multi-pattern situations.

## What is the core method / protocol?

- Define a library of psychological patterns as structured objects:
  - 100 personality traits (Goldberg’s Big Five markers)
  - 144 social-cognitive patterns (biases, social influence mechanisms, motivation, etc.)
- Build a synthetic scenario dataset:
  - 11,359 scenarios
  - each scenario includes 2–5 patterns that may reinforce, conflict, or modulate each other
  - multi-turn conversations include *inner thoughts* (brackets), *actions* (parentheses), and *dialogue*
- Train via supervised fine-tuning (e.g., HumanLLM-8B/32B based on Qwen3-8B/32B) on the generated conversations.
- Evaluate with **dual-level checklists**:
  - Pattern-level checklist: 15 universal behavioral indicators per pattern
  - Scenario-level checklist: 2–6 items per target character capturing expected behavior under the specific multi-pattern configuration

## What are the key metrics?

- Checklist-based judge scoring at:
  - pattern-level fidelity
  - scenario-level / emergent multi-pattern dynamics
- “Human alignment” correlation (reported as r=0.91) between their automatic evaluation and human assessment.
- They also analyze how holistic metrics can correlate with social desirability rather than simulation faithfulness.

## What are the main results?

- HumanLLM-8B reportedly beats Qwen3-32B on **multi-pattern dynamics** evaluation despite much smaller parameter count.
- Their analysis suggests multi-pattern checklisting surfaces failure modes that are hidden by single-number “anthropomorphism/human-likeness” measures.

## How is this similar to GALILEO?

- Directly targets **persona/personality stability** and highlights **drift / inconsistency** in multi-turn settings.
- Emphasizes evaluation protocols that go beyond single-turn prompts (multi-turn scenarios; interaction effects).
- Uses *structured rubrics/checklists* to make evaluation more audit-able and decomposable (conceptually aligned with GALILEO’s desire for reproducible, failure-mode-revealing metrics).

## How is this different from GALILEO?

- HumanLLM is primarily about **anthropomorphic role-play fidelity** (psychological pattern simulation) and includes **training** (SFT) to improve it.
- GALILEO is a **truth-grounded** multi-turn robustness benchmark (survival / turn-of-failure / recovery under persona pressure) with automatic scoring anchored to ground-truth answers.
- HumanLLM’s checklists are about *behavioral indicators* of psychological patterns; GALILEO’s protocol measures *answer consistency under adversarial conversational pressure*.

## Where GALILEO is stronger / cleaner (if true)

- Clear separation between “socially pleasing” responses and **objective correctness** via ground-truth tasks (less risk of conflating fidelity with desirability).
- Metrics like survival curves, turn-of-failure, and recovery are simple, interpretable, and comparable across tasks/models.

## Where GALILEO is weaker / needs to improve

- Does not explicitly model or evaluate **interacting psychological patterns** (reinforcement/conflict/modulation) as first-class factors.
- Could benefit from more rubric-like decomposition of *why* a flip happened (beyond persona category), analogous to their pattern-level indicators.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite HumanLLM as evidence that (a) personality/behavioral traits interact, and (b) multi-turn evaluation should separate fine-grained fidelity from holistic “good behavior” metrics.
- [ ] Consider adding an analysis layer that tags flips with a small checklist (e.g., deference cues, authority compliance, spotlight/embarrassment cues) to better connect to psychology-pattern literature without abandoning ground-truth scoring.

## Quotes / details to potentially cite

- Dataset scale: “244 patterns” (100 personality traits + 144 social-cognitive patterns) grounded in “~12,000” papers; “11,359” scenarios; 2–5 patterns per scenario.
- Evaluation idea: dual-level checklists (pattern-level universal indicators + scenario-level expected tendencies) and claim that holistic metrics conflate simulation accuracy with social desirability.
