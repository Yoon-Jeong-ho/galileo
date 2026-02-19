# Self-Correcting Large Language Models: Generation vs. Multiple Choice

- Year: 2025
- Venue: arXiv
- Authors: Hossein A. Rahmani; Satyapriya Krishna; Xi Wang; Mohammadmehdi Naghiaei; Emine Yilmaz
- URL: https://arxiv.org/abs/2511.09381
- BibTeX key (if we add it): rahmani2025selfcorrecting
- Tags: self-correction, iterative-refinement, semantic-drift, multiple-choice, evaluation

## One-sentence takeaway

Iterative self-correction behaves differently depending on output space: open-ended generation shows rapid early gains but can drift/degenerate with more rounds, while multiple-choice is stable but often cannot “flip” from an initially wrong option.

## What problem does it solve?

- Clarifies when/why popular “self-correction” (self-reflection / iterative revision) helps or hurts, disentangling effects of *task/output format* (free-form generation vs. fixed-option multiple choice).
- Provides empirical evidence that “more reflection rounds” is not universally beneficial and can introduce semantic drift in generative settings.

## What is the core method / protocol?

- Head-to-head evaluation of iterative self-correction across two parallel task formulations:
  - **Open-ended generation**: model produces a text answer, then is prompted to review and revise for up to ~5 rounds.
  - **Multiple-choice selection**: model selects among fixed options (via logits); self-correction is implemented by generating rationales / reflection and recomputing option scores across rounds.
- Benchmarks:
  - **DisambiguationQA** (can be posed as MC with 4 options; also cast to generation by asking for the referent).
  - **tinyTruthfulQA** (subset of TruthfulQA; evaluated in generation and MC variants).
- Models span sizes/families (as reported): SmolLM2-1.7B, Qwen2.5-3B, Llama-3.1-8B, Qwen2.5-14B, DeepSeek-R1-Distill-Llama-8B, Gemini-2.0-Flash.
- Analysis focuses on **accuracy per iteration** and **flip dynamics**:
  - “correct flips” = wrong → right across rounds
  - “incorrect flips” = right → wrong across rounds

## What are the key metrics?

- Accuracy vs. self-correction iteration (0..5).
- Counts/rates of correct vs. incorrect flips across iterations (proxy for stability vs. over-correction / drift).

## What are the main results?

- **Generation**: improvement is typically **front-loaded** (1–2 rounds), then **plateaus or declines** as more rounds increase the risk of semantic drift and new mistakes.
- **Multiple-choice**: accuracy increases **gradually/steadily**, and answers are **stable** (few flips), but:
  - If the initial choice is wrong, later rounds **rarely flip** to the correct option (“logit inertia”).
- Dataset-dependent pattern:
  - On **DisambiguationQA** (hard), even large models plateau around ~50% (MC) and substantially lower for generation (text mentions <~20% in their setup), and iterative gains are modest.
  - On **tinyTruthfulQA** (easier), generation can be relatively high (~60–90% range reported), and MC ~50–80%.

## How is this similar to GALILEO?

- Directly relevant to any GALILEO-style iterative/agentic loop that alternates between:
  - open-ended plan/explanation generation, and
  - discrete constrained decisions (tool/action selection),
  where “self-correction” could be applied in both modes.
- Highlights the same practical tension: iteration can improve early but may later introduce drift or regressions.

## How is this different from GALILEO?

- This is an *evaluation/characterization* paper, not a new GALILEO-like training or inference framework.
- Studies relatively simple iterative prompting loops on QA-style benchmarks, not full agent trajectories with tool feedback, environment state, or long-horizon credit assignment.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses explicit state/constraints/checks (e.g., verifier/tool feedback, structured action spaces), it can potentially avoid the “late-round generative drift” observed here.

## Where GALILEO is weaker / needs to improve

- If GALILEO relies on repeated free-form self-revision without strong grounding/verifiers, it may inherit the same instability: later iterations can turn correct intermediate reasoning into incorrect outputs.

## Action items for GALILEO (experiments / method / writing)

- [ ] In GALILEO experiments, report **accuracy vs. iteration** curves and **flip rates** (wrong→right, right→wrong) to quantify “over-correction”.
- [ ] When comparing agent designs, explicitly separate **free-form** steps (plans/explanations) from **discrete** steps (action selection) and test self-correction in each.
- [ ] Add/strengthen drift controls: stopping criteria, consistency checks, external verification, or constraints that prevent off-topic revisions.
- [ ] In writing, cite this paper to justify why “more reflection rounds” can be harmful in generation, and why constrained decisions can be stable but hard to change once mistaken.

## Quotes / details to potentially cite

- They frame the key contrast as: open-ended generation benefits from flexible re-interpretation/compositional refinement, while multiple-choice has clearer boundaries but is limited by the provided options.
- Qualitative dynamics summary (paraphrase): generation gains early then risks semantic drift; multiple-choice is stable but suffers from inertia when the initial option is wrong.
- Benchmarks used for parallel format comparisons: DisambiguationQA and tinyTruthfulQA.
