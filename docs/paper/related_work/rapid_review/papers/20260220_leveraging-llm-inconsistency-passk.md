# Leveraging LLM Inconsistency to Boost Pass@k Performance

- Year: 2025
- Venue: arXiv
- Authors: Uri Dalal; Meirav Segal; Zvika Ben-Haim; Dan Lahav; Omer Nevo
- URL: https://arxiv.org/abs/2505.12938
- BibTeX key (if we add it): Dalal2025InconsistencyPassAtK
- Tags: self-consistency, pass@k, inconsistency, prompt-variation, ensembling, coding, cybersecurity, reasoning-models

## One-sentence takeaway

Instead of sampling k solutions to the same prompt, generate k *equivalent prompt variants* and sample one solution per variant; due to prompt-sensitivity/inconsistency, this can yield higher Pass@k.

## What problem does it solve?

- Standard Pass@k is typically improved by sampling multiple responses to the *same* task prompt (“repeater” style), but LLMs are brittle: small, semantically-equivalent prompt variations can swing success probability.
- The paper asks: can we *use* this inconsistency to improve Pass@k in domains where solutions can be automatically verified (e.g., coding, some cybersecurity tasks)?

## What is the core method / protocol?

- Proposed agent: **Variator**.
  - Step 1: Use an LLM to generate **k equivalent variants** of the original task (paraphrase + change theme/backstory/notation/terminology; keep I/O spec for coding tasks).
  - Step 2: For each variant, generate **one** candidate solution.
  - Compare against baseline **Repeater**: generate **k** candidate solutions to the original prompt.
- Theory: a probabilistic model of the “inconsistency effect” shows Variator can improve Pass@k even if the *average* single-try solve rate over variants is not higher than the original; the gain comes from occasionally producing a “lucky” variant with much higher solve probability.
- Experiments:
  - Public coding benchmark: APPS.
  - Also studies persistence of inconsistency in **frontier reasoning models** on coding + cybersecurity challenges (the paper text mentions OpenAI o3-mini and Claude 3.7 Sonnet extended thinking mode), with automatically-testable success criteria.

## What are the key metrics?

- **Pass@k** (success if any of k attempts is correct).
- For inconsistency demonstrations: per-variant success rates across many variants of the same underlying task.

## What are the main results?

- Empirically, Variator outperforms the Repeater baseline on APPS (and a private dataset of task variants) according to the paper.
- The “inconsistency effect” remains present even for advanced reasoning-focused models, across both coding and cybersecurity tasks.
- Variant generation is mostly reliable but not perfect; they report expert checks and note a small fraction of coding variants are non-equivalent (paper mentions 6% for coding variants in their process).

## How is this similar to GALILEO?

- If GALILEO uses multiple candidate generations / multi-sample evaluation (any Pass@k-style regime), this paper is a direct argument that **diversity over prompts** (equivalent reformulations) can be a better lever than diversity over stochastic decoding alone.
- The “agent framing” (generate variants, then solve) is compatible with modular pipelines.

## How is this different from GALILEO?

- Focus is specifically on **prompt-variant ensembling** to improve *Pass@k*; not primarily about new model training, retrieval, or formal verification.
- Assumes a setting where solutions can be automatically checked (tests / flags), which may not cover all GALILEO target tasks.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has stronger guarantees about equivalence-preservation (or uses programmatic transformations / formal constraints), it could avoid the “non-equivalent variant” issue that requires expert filtering here.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently samples k solutions for a fixed prompt, it may be leaving performance on the table versus prompt-variant strategies.
- If GALILEO lacks a robust “prompt variant generator” with equivalence checks, it may not exploit inconsistency effectively.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a baseline/ablation: **Repeater vs Variator** for any GALILEO evaluation that reports Pass@k or best-of-n.
- [ ] Implement an equivalence-preserving variant generator (for structured tasks, constrain to preserve I/O spec; for others, define task-specific invariants).
- [ ] Add an experiment: measure **variance of success across variants** for a fixed task to quantify inconsistency, and correlate with gains from Variator.
- [ ] Consider adding a lightweight “variant quality filter” (self-check + heuristic constraints) to reduce non-equivalent variants.

## Quotes / details to potentially cite

- “Rather than view [inconsistency] as a drawback, … leveraging models’ inconsistency to boost Pass@k performance.”
- Variator description: “generates k variants of a given task and submits one candidate solution for each one.”
- Persistence claim: “inconsistency persists even in frontier reasoning models across coding and cybersecurity domains.”
