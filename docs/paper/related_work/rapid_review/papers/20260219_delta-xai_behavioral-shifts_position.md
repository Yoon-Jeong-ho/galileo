# Position: Explaining Behavioral Shifts in Large Language Models Requires a Comparative Approach

- Year: 2026
- Venue: arXiv (Position)
- Authors: Francesco Giannini et al. (arXiv “From:” metadata)
- URL: https://arxiv.org/abs/2602.02304
- BibTeX key (if we add it): giannini2026deltaxai (suggested)
- Tags: behavioral-shifts, interpretability, comparative-analysis, xai

## One-sentence takeaway

Proposes “Comparative XAI” (Δ-XAI) as the right framing for explaining *model behavioral shifts* by attributing differences between checkpoints (pre/post intervention), not explaining each checkpoint in isolation.

## What problem does it solve?

- Behavioral shifts (emergent abilities/misalignments, post-RLHF changes, post-finetune changes, etc.) are governance- and safety-relevant, but most XAI methods are single-checkpoint and don’t directly justify *what changed* across versions.
- For safety/robustness work, we often need *comparative* claims: “this intervention caused increased sycophancy / reduced resistance / more drift,” and what internal changes mechanistically explain it.

## What is the core method / protocol?

- A position / framework paper introducing **Δ-XAI (Comparative XAI)**:
  - Target of explanation is the **delta** between a *reference* model and an *intervened* model.
  - Provides desiderata for comparative explainability methods (framed as design requirements).
  - Sketches possible pipelines and gives an example Δ-XAI experiment (details not fully captured in the truncated HTML excerpt).
- Key conceptual move: explainability should be **intervention-centric** (scaling, fine-tuning, RL, ICL) and **difference-centric**.

## What are the key metrics?

- Not primarily metric-driven (framework paper).
- Implied evaluation axes for any Δ-XAI method:
  - robustness / reproducibility of attributed differences across seeds and prompt paraphrases
  - localization / specificity of explanation to the shift-causing components
  - causal testability (e.g., patching/ablation that reverses or induces the shift)

## What are the main results?

- Argues standard XAI is “structurally ill-suited” for explaining checkpoint-to-checkpoint behavioral shifts.
- Introduces Δ-XAI framing + desiderata; motivates via examples including sycophancy, alignment changes, tool-use changes, and other emergent phenomena.

## How is this similar to GALILEO?

- GALILEO’s core concerns (multi-turn robustness under pressure, sycophancy/persuasion resistance, belief revision vs drift) are exactly the kind of **behavioral properties that shift** after training/eval interventions.
- Provides a vocabulary for explaining *why* GALILEO (or baselines) change behavior across:
  - instruction-tuning variants
  - anti-sycophancy training
  - memory/long-context tweaks
  - tool-use scaffolds

## How is this different from GALILEO?

- Not an evaluation benchmark or training method; it’s a **conceptual + methodological framing** for interpretability across checkpoints.
- Doesn’t directly propose multi-turn stress tests; instead it focuses on explaining *mechanisms of shifts* once identified.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides concrete multi-turn “pressure” protocols and quantitative robustness metrics, it will be more directly actionable for measuring the shift.
- GALILEO can ground Δ-XAI desiderata in a specific failure mode taxonomy (e.g., sycophantic drift vs justified belief revision).

## Where GALILEO is weaker / needs to improve

- If GALILEO claims “we reduce drift / sycophancy” without mechanistic support, this paper is a reminder that *comparative causal explanations* are increasingly expected (auditing/governance angle).
- GALILEO should anticipate “behavioral shift” critiques: are improvements stable across checkpoints / seeds / paraphrases, and are they attributable to intended mechanisms?

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “Δ-analysis” section framing: when we introduce any intervention, explicitly define the **behavioral shift of interest** (pre vs post) and how we will explain it.
- [ ] For at least one key GALILEO improvement (e.g., reduced persuasion/sycophancy), include a **comparative mechanistic probe**:
  - activation patching / causal mediation-style tests that reverse/induce the shift
  - representation similarity / probe comparisons, but with explicit caveats about causality
- [ ] In evaluation: report **stability of the shift** across seeds, prompt paraphrases, and multi-turn variants (connects to “robustness” desiderata).
- [ ] Writing: cite this as motivation for why we compare behaviors across training interventions and why black-box eval alone is insufficient for explaining shifts.

## Quotes / details to potentially cite

- “We take the position that behavioral shifts should be explained comparatively: the core target should be the intervention-induced shift between a reference model and an intervened model, rather than any single model in isolation.” (arXiv HTML abstract/introduction)
- Classic XAI methods are “structurally ill-suited to justify what changed internally across different checkpoints and which explanatory claims are warranted about that change.” (abstract)
