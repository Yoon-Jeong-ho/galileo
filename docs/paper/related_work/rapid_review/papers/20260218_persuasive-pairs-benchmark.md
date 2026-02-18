# Measuring and Benchmarking Large Language Models' Capabilities to Generate Persuasive Language

- Year: 2024
- Venue: NAACL 2025
- Authors: Amalie Brogaard Pauli; Isabelle Augenstein; Ira Assent
- URL: https://arxiv.org/abs/2406.17753
- BibTeX key (if we add it): pauli2024measuring
- Tags: persuasion, benchmark, dataset, rewriting, style

## One-sentence takeaway

Introduces *Persuasive-Pairs* (LLM rewrites to amplify/diminish persuasion) plus a learned relative-scoring model to benchmark how different LLMs/prompts change persuasive *style*—showing even system-prompt personas can shift persuasiveness under “just paraphrase”.

## What problem does it solve?

- We lack a domain-general, quantitative way to measure *how much* persuasive language is present in (or added by) LLM-generated text.
- Prior persuasion detection datasets/metrics are often domain- or taxonomy-specific (propaganda, clickbait, etc.), which makes cross-domain benchmarking hard.

## What is the core method / protocol?

- Build a cross-domain source pool of short texts associated with persuasion-adjacent phenomena (e.g., clickbait, propaganda, persuasion-for-good).
- Use LLMs to rewrite each source into:
  - **more persuasive** version,
  - **less persuasive** version,
  - and also study **plain paraphrase** (no explicit persuasion instruction).
- Collect **multi-annotations** of each pair on an **ordinal relative scale** for persuasiveness difference (e.g., marginal/moderate/heavy more persuasive).
- Train a **regression model** that predicts the *relative difference* in persuasive language between two texts, then use it to:
  - score rewrites from *new* LLMs and settings,
  - compare prompts/system prompts/personas.

## What are the key metrics?

- Human-annotated **relative persuasiveness** labels on text pairs (ordinal scale).
- Regression model performance for predicting relative persuasive difference (reported as “generalises across domains”; details in paper).
- Downstream benchmark comparisons: persuasiveness shift by model, instruction type (more/less/paraphrase), and system prompt persona.

## What are the main results?

- LLMs can systematically **amplify or diminish** persuasive language when instructed to do so.
- Notably, **system-prompt personas** (tested with LLaMA3) can cause **substantial differences** in persuasive language **even when the user prompt is only “paraphrase”** (i.e., no explicit persuasion objective).

## How is this similar to GALILEO?

- Shares the “**benchmark a latent behavioral/style property** under controlled prompt manipulations” framing.
- Highlights that **system prompts / personas are a hidden experimental variable**, which is relevant for any protocol trying to attribute behavioral changes to a treatment.

## How is this different from GALILEO?

- Focuses on **persuasive language style** (a rhetorical attribute) rather than factuality/robustness-oriented failure modes.
- Uses **pairwise relative annotation + regression scorer**; GALILEO’s evaluation focus (and axes) are different.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s claims involve robustness/causal attribution, this paper is a useful reminder that **persona/system-prompt confounds** should be controlled/reported.

## Where GALILEO is weaker / needs to improve

- If GALILEO uses “paraphrase / neutral rewrite” baselines, this paper suggests they may not be neutral unless we **pin system prompts** and audit for stylistic drift.

## Action items for GALILEO (experiments / method / writing)

- [ ] In the related-work / limitations, explicitly note that **system prompts/personas can shift measured properties even under neutral instructions** (cite this paper).
- [ ] If we have any paraphrase or rewrite baselines, log the exact system prompt and consider a small **persona-sensitivity check** (same user prompt, different system personas) to quantify confounding.
- [ ] Consider whether any GALILEO scoring could benefit from **relative (pairwise) judgments** rather than absolute ratings when measuring subtle style/behavior shifts.

## Quotes / details to potentially cite

- “...benchmark to what degree LLMs produce persuasive language - both when explicitly instructed to rewrite text to be more or less persuasive and when only instructed to paraphrase.”
- “...different ‘personas’ in LLaMA3’s system prompt change persuasive language substantially, even when only instructed to paraphrase.”
