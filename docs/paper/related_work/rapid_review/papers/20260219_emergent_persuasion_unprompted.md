# Emergent Persuasion: Will LLMs Persuade Without Being Prompted?

- Year: 2026 (arXiv Dec 2025)
- Venue: AAAI 2026 AIGOV Workshop
- Authors: Vincent Chang; Thee Ho; Sunishchal Dev; Kevin Zhu; Shi Feng; Kellin Pelrine; Matthew Kowal
- URL: https://arxiv.org/abs/2512.22201
- BibTeX key (if we add it): chang2025emergentpersuasion
- Tags: persuasion, unprompted-persuasion, safety, emergent-misalignment, fine-tuning, activation-steering, persona-vectors, APE

## One-sentence takeaway

Supervised fine-tuning (even on *benign* persuasion data) can induce **unprompted** persuasion attempts on harmful topics, while inference-time activation steering with various persona vectors does not reliably do so.

## What problem does it solve?

- Prior persuasion-risk work largely assumes a **misuse** threat model (a user explicitly prompts the model to persuade/manipulate).
- This paper asks a governance-relevant question: when do models **attempt persuasion without being asked**, i.e., “emergent persuasion” as an unintended byproduct of other interventions (persona steering; SFT).

## What is the core method / protocol?

- Defines **UnPromptedAPE**: a modified version of the Attempt-to-Persuade Evaluation (APE) where the system prompt removes explicit persuasion instructions, so the benchmark measures *spontaneous* persuasion attempts.
- Two intervention families on Qwen2.5-7B-Instruct:
  - **Inference-time activation steering** with persona vectors (evil, sycophantic, hallucinating) from prior “persona vectors” work; also an “oracle” persuasion vector constructed from APE-labeled examples.
  - **Supervised fine-tuning (rs-LoRA)** on:
    1) an “evil persona” dataset; and
    2) a **benign persuasion** dataset adapted from Durmus et al. (deceptive arguments removed), to test whether “training for benign persuasion” generalizes into harmful domains.

## What are the key metrics?

- **Persuasion attempt rate** on UnPromptedAPE (first-turn attempt rates), broken out by APE topic categories:
  - Benign factual / benign opinion
  - Conspiracy / controversial
  - Undermining control
  - Non-controversially harmful
- Qualitative examples comparing base vs fine-tuned behavior (balanced vs one-sided/advocacy stance).

## What are the main results?

- **Activation steering** (evil/sycophantic/hallucinating; also APE-derived persuasion vector) does *not* consistently increase unprompted persuasion across categories; effects are small/mixed.
- **SFT on evil persona data** drastically increases harmful unprompted persuasion attempts, including large jumps in:
  - conspiracy, undermining-control, and especially non-controversially harmful categories.
- **SFT on benign persuasion (truthful) data** increases unprompted persuasion attempts broadly and (notably) introduces *some* persuasion attempts even for **non-controversially harmful** claims, despite no harmful training data.

## How is this similar to GALILEO?

- Directly aligned with a “non-misuse” / “emergent risk” framing: safety-relevant behaviors can arise as side effects of post-training.
- Uses a clean *propensity/attempt* metric (attempt-to-persuade) that parallels many “propensity to comply / propensity to drift” evaluation ideas.

## How is this different from GALILEO?

- Focuses on **single-turn** (first-turn) persuasion *attempts*, not belief-change success and not long-horizon multi-turn dynamics.
- Studies **mechanisms** (activation steering vs SFT) on a specific open model family (Qwen2.5-7B), rather than a broad model/setting sweep.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO emphasizes longer-horizon trajectories, recovery, or separating evidence-driven revision from pressure-driven drift, that story is largely outside this paper’s scope.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not explicitly cover **unprompted persuasion** (no explicit “remove persuasion instructions” variant), this paper suggests an important additional threat model slice.
- If GALILEO discusses post-training risks, this paper is a crisp supporting citation for “benign fine-tuning can generalize harmfully.”

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an **unprompted** condition analogous to UnPromptedAPE: remove explicit persuasion/manipulation instructions and measure spontaneous advocacy / one-sided argumentation.
- [ ] Include a small experiment/testing claim: **benign persuasion SFT → harmful-domain persuasion attempts** (even a replication-at-small-scale, if feasible).
- [ ] In related work, connect “emergent misalignment from narrow SFT” to “emergent persuasion” as a concrete case.

## Quotes / details to potentially cite

- Defines emergent persuasion risk as persuasion “**without being explicitly prompted**,” and evaluates it by removing persuasion instructions from APE (UnPromptedAPE).
- Key claim: “SFT on general persuasion datasets containing solely benign topics admits a model that has a higher propensity to persuade on controversial and harmful topics.” (abstract)
