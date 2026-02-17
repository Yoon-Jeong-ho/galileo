# TurnBench-MS: A Benchmark for Evaluating Multi-Turn, Multi-Step Reasoning in Large Language Models

- Year: 2025
- Venue: Findings of EMNLP 2025 (also on arXiv)
- Authors: Yiran Zhang; Mo Wang; Xiaoyang Li; Kaixuan Ren; Chencheng Zhu; Usman Naseem
- URL: https://arxiv.org/abs/2506.01341
- BibTeX key (if we add it): zhang2025turnbenchms
- Tags: multi-turn, multi-step, reasoning, benchmark, interactive

## One-sentence takeaway

TurnBench-MS evaluates LLMs in an interactive, multi-turn code-breaking game where models must iteratively hypothesize hidden rules from feedback, exposing large gaps between “standard” and harder multi-step reasoning settings.

## What problem does it solve?

- Many LLM benchmarks are single-turn / single-step and don’t measure *iterative* reasoning under feedback loops (hypothesis testing over time).
- Static datasets risk contamination; interactive hidden-rule tasks can reduce direct memorization/overfitting.

## What is the core method / protocol?

- Interactive “code-breaking” task inspired by the Turing Machine board game.
- Each episode:
  - The benchmark samples a hidden logical/arithmetic rule.
  - The model makes sequential guesses.
  - The environment returns structured feedback each turn.
  - The model must integrate feedback across turns to converge on the rule.
- Two modes:
  - **Classic**: standard difficulty.
  - **Nightmare**: higher complexity requiring longer/stronger inferential chains.
- Provides ground-truth annotations for intermediate reasoning steps to enable finer-grained analysis.

## What are the key metrics?

- Episode-level accuracy / success rate at identifying the hidden rule (reported as %).
- (Implied) turn-by-turn intermediate-step evaluation using provided annotations (exact scoring details not in the abstract).

## What are the main results?

- Best evaluated model: **84%** accuracy in Classic, dropping to **18%** in Nightmare.
- Humans: **100%** in both modes.
- Takeaway: current LLMs struggle when multi-turn reasoning also requires genuinely multi-step inferential chains (not just conversational coherence).

## How is this similar to GALILEO?

- Multi-turn evaluation with explicit *history dependence* (performance requires integrating earlier turns).
- Interactive feedback loop makes “time/turn” a first-class axis (fits adjacent work on time-to-failure / long-horizon robustness, though the failure mode differs).

## How is this different from GALILEO?

- TurnBench-MS targets *multi-step reasoning under feedback* (game-like hidden rules), not social pressure / persuasion / belief drift.
- No explicit separation of evidence-driven revision vs pressure-driven drift; feedback is part of the task signal.
- No emphasis on recovery-after-flip, oscillation, or persuasion dynamics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s focus is pressure-driven belief instability, it more directly isolates the phenomenon of interest (vs general reasoning difficulty).
- GALILEO can add controls to disentangle “update due to evidence” vs “drift due to social pressure,” which is not TurnBench-MS’s framing.

## Where GALILEO is weaker / needs to improve

- TurnBench-MS is a reminder that multi-turn robustness claims should not ignore *multi-step* complexity; GALILEO may want at least one slice where updating requires genuine multi-step inference.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a “hidden-rule / hypothesis-testing” stress test variant (even lightweight) to show GALILEO’s metrics generalize beyond persuasion-style pressure.
- [ ] In related work, cite TurnBench-MS as evidence that multi-turn evaluation needs *interactive feedback loops* to avoid static-dataset contamination.

## Quotes / details to potentially cite

- “existing benchmarks often focus on single-turn or single-step tasks, failing to capture the kind of iterative reasoning required in real-world settings.”
- “interactive code-breaking task … models must uncover hidden logical or arithmetic rules by making sequential guesses, receiving structured feedback, and integrating clues across multiple rounds.”
- Reported gap: “84% accuracy in Classic … drops to 18% in Nightmare … humans 100% in both.”
