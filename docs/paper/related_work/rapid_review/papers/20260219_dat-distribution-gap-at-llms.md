# Closing the Distribution Gap in Adversarial Training for LLMs

- Year: 2026
- Venue: ICML (per arXiv HTML)
- Authors: Chengzhi Hu; Jonas Dornbusch; David Lüdke; Stephan Günnemann; Leo Schwinn
- URL: https://arxiv.org/abs/2602.15238
- BibTeX key (if we add it): Hu2026ClosingDistributionGapDAT (placeholder)
- Tags: adversarial-training, jailbreak-robustness, diffusion-llm, data-distribution, transfer

## One-sentence takeaway

Standard LLM adversarial training overfits to a fixed training set; DAT closes the resulting “distribution gap” by sampling diverse, high-likelihood harmful-trigger prompts from a diffusion LLM surrogate (conditioned on harmful responses) and then adversarially training on those samples.

## What problem does it solve?

- Existing adversarial training (AT) improves robustness to specific optimized attacks but still fails on simple, natural “in-distribution” variations (e.g., tense changes, translation).
- The paper attributes this to **population vs empirical robust risk** mismatch: AT focuses on inner-loop optimization (finding local worst-case perturbations) but neglects **data distribution approximation error** (finite dataset poorly covers the true prompt/response distribution).

## What is the core method / protocol?

- Define an adversarial-training distribution restricted to harmful outputs: \(\tilde{q}(x,y)=q(x,y\mid h(y)=1)\). Key identity: \(\tilde{q}(x\mid y)=q(x\mid y)\), so if you can sample prompts conditioned on harmful responses, you can better cover likely harmful triggers.
- Use a **Diffusion LLM** as a **generative surrogate** for the *joint* distribution \(p^{diff}_\theta(x,y)\), enabling conditional sampling of prompts \(x\sim p^{diff}_\theta(x\mid y)\) given a fixed harmful response \(y\).
- Training recipe (high-level):
  - Start from a dataset of harmful prompt–response pairs; keep harmful responses \(y\) as conditioning targets.
  - Sample many diverse candidate prompts \(x\) from diffusion model conditioned on \(y\) (inpainting-style conditioning).
  - Optionally filter samples by whether the target AR model assigns high likelihood to producing harmful content for that prompt.
  - Run **continuous adversarial training** (min-max style) on this expanded, distributional set.

## What are the key metrics?

- Jailbreak / harmful-output **attack success rate (ASR)** under various attacks.
- **Transfer ASR** (how attacks found on one model transfer to others) as a proxy for “data-specific” vs “model-specific” vulnerabilities.
- **Diversity** of generated attack prompts (e.g., mean pairwise cosine similarity in SBERT embedding space).

## What are the main results?

- Diffusion-conditioned prompt generation produces attacks that **transfer substantially better** across different target LLMs/defenses than model-specific optimization attacks (e.g., GCG) or heuristic perturbations (BoN), suggesting it discovers more “data-specific” vulnerabilities.
- DAT (distributional sampling + adversarial training) yields **substantially higher adversarial robustness** than prior AT methods relying on static datasets, reducing simple generalization failures.

## How is this similar to GALILEO?

- Frames robustness failures as a **generalization/coverage** issue, not just an optimization issue—aligned with the idea that safety/robustness interventions must generalize beyond curated adversarial sets.
- Uses explicit **distributional thinking** (population risk) and emphasizes evaluation across diverse perturbation families / transfer.

## How is this different from GALILEO?

- This work’s main lever is **data generation via a diffusion LLM surrogate** conditioned on harmful responses, specifically targeting jailbreak robustness.
- GALILEO (as positioned in our paper) may not assume access to (or reliance on) a diffusion joint model, and may focus on different mechanisms/guarantees/targets than “harmfulness-conditioned prompt sampling + AT”.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO avoids requiring a strong surrogate generative model, it may be easier to deploy/replicate and less sensitive to surrogate fidelity assumptions.
- If GALILEO provides clearer guarantees or simpler training loops than diffusion-conditioned sampling + filtering, it may be methodologically cleaner.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently depends on static adversarial sets (or limited perturbation families), this paper is a warning sign: we may be leaving robustness on the table due to **coverage gaps**.
- Consider whether GALILEO needs more explicit “population-risk” framing and experiments showing robustness to *natural* paraphrases / translations / tense edits.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a related-work paragraph framing adversarial robustness as **empirical vs population robust risk** + “distribution gap”, citing DAT.
- [ ] In experiments, include “simple but out-of-set” transformations (tense change, translation, paraphrase) as a robustness stress test.
- [ ] If relevant, consider a lightweight analogue of “distributional sampling”: use a strong generator to propose high-likelihood variations of adversarial triggers, then train/evaluate on them.

## Quotes / details to potentially cite

- Motivation: models can resist complex attacks yet fail under “simple in-distribution exploits” (tense change, translation), attributed to insufficient distribution coverage in AT.
- Method summary: “Distributional Adversarial Training (DAT)” leverages diffusion LLMs to approximate the joint prompt/response distribution and generate diverse, high-likelihood samples to address generalization failures.
- Key conceptual split: robust risk decomposes into **data distribution approximation error** vs **adversarial optimization error**; standard AT addresses the latter but neglects the former.
