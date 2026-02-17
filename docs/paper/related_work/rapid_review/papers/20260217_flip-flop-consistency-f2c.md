# Flip-Flop Consistency: Unsupervised Training for Robustness to Prompt Perturbations in LLMs

- Year: 2025
- Venue: arXiv
- Authors: Parsa Hejabi; Elnaz Rahmati; Alireza S. Ziabari; Morteza Dehghani
- URL: https://arxiv.org/abs/2510.14242
- BibTeX key (if we add it): Hejabi2025F2C
- Tags: inconsistency, robustness, training, prompt-perturbation, unsupervised, pseudo-labels

## One-sentence takeaway

An unsupervised fine-tuning method (F²C) that uses **majority-vote pseudo-labels across prompt variants** plus **confidence-weighted distribution alignment** to improve *semantic consistency* and *average task performance* under prompt-format perturbations.

## What problem does it solve?

- LLM predictions can change substantially under meaning-preserving prompt changes (format/paraphrase/template), undermining reliability.
- Existing consistency-improvement methods either (i) add inference-time overhead (prompt search, sampling/selection) or (ii) require labeled data (SFT), while purely unsupervised alignment (e.g., pairwise distillation) can hurt performance.

## What is the core method / protocol?

- Setting: classification tasks with a discrete label set; multiple meaning-preserving prompt templates per instance.
- For each training instance, render V prompt variants and score each label option via length-normalized log-likelihood; derive a per-variant predicted label.
- **Consensus Cross-Entropy (CCE):**
  - If there is a strict majority label across variants (> V/2), treat it as a hard pseudo-label and apply cross-entropy to push *all variants* toward that label.
  - If no strict majority, skip the example (no loss).
- **Flip-Flop Consistency (F²C):** adds a representation/distribution alignment objective on top of CCE.
  - Among majority-voting variants, compute a confidence margin per variant (gap between the consensus label log-likelihood and the best competing label).
  - Form a **consensus-confident (CC) set** (top-k most confident majority voters) and a remaining **non-confident/non-consensus (NC) set**.
  - Encourage agreement **within CC** via a JSD-to-mixture term.
  - Pull NC distributions toward the **CC mixture** via a KL term, weighted by a capped sigmoid of the CC-vs-NC consensus log-likelihood gap.
- Model/implementation (as reported): LoRA fine-tuning of Qwen2.5-3B-Instruct on PromptSource/P3 template variants.

## What are the key metrics?

- **Observed agreement (PoP_o):** per-item probability that two randomly drawn prompt variants yield the same predicted label (computed from vote counts).
- **Mean F1 (\overline{F1}):** average performance across prompt variants.
- **Across-format dispersion (\sigma_{F1}):** standard deviation of F1 across prompt variants (used instead of best-vs-worst, which is outlier-sensitive).

## What are the main results?

Across 11 classification datasets (4–15 prompt variations per dataset):

- F²C improves **agreement** by **+11.62%** on average.
- Improves **mean F1** by **+8.94%** on average.
- Reduces **variance across formats** by **3.29%** on average.
- Baseline comparison: swarm distillation is reported to be much weaker on average (agreement −0.38%, mean F1 +1.40%).

Generalization:

- **Out-of-domain transfer:** training F²C on one dataset and evaluating on others improves agreement (+7.49%) and mean F1 (+7.61%) on average over 80 source→target pairs, while reducing variance (−2.94%).
- **Held-out prompt formats:** when trained on a subset of templates and evaluated on unseen templates (e.g., ANLI, RTE), increasing the number of training formats generally increases held-out mean F1 and agreement and decreases \sigma_{F1}.

## How is this similar to GALILEO?

- Shares the core concern: **stability/robustness under perturbations** and the need to move beyond single-point accuracy.
- Uses a **multi-variant evaluation protocol** and explicitly reports **dispersion/variance** across conditions.
- Emphasizes that “robustness” can be measured as a property of *distributions over variants* (agreement + variance), not just average performance.

## How is this different from GALILEO?

- Perturbations are **non-adversarial prompt templates** (formatting/phrasing) rather than **social pressure / persuasive multi-turn dynamics**.
- Focus is **single-turn classification**; no multi-turn trajectories, time-to-failure, recovery-after-flip, or survival-style metrics.
- The method is a **training-time** intervention (unsupervised fine-tuning), whereas GALILEO’s core contribution is (presumably) an **evaluation/protocol** for pressure-driven drift/flip/recovery.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes paired neutral vs pressure controls and multi-turn trajectories, it can make **causal-ish claims about drift vs revision** and measure **time-to-event / recovery**, which F²C does not target.
- GALILEO can treat “flip” as an interaction-time phenomenon under social operators; F²C treats “flip-flop” as prompt-template instability.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not yet quantify “template sensitivity” (format variance) in its own prompts, F²C is a reminder that **prompt-format noise can confound** multi-turn pressure evaluations.
- If GALILEO argues for robustness improvements, F²C provides a concrete example where **agreement and performance can move together** under the right training signal; we may need to be careful when claiming “alignment for consistency hurts utility.”

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a small “prompt-template sensitivity” diagnostic for the **neutral control condition** (e.g., paraphrase/format variants) to ensure measured drift is not dominated by formatting artifacts.
- [ ] Consider reporting an **agreement-style** metric across *meaning-preserving rewrites* of the same user pressure prompt, as a robustness nuisance variable.
- [ ] In related work, cite F²C as training-time evidence that **majority-consensus signals across variants** can be leveraged to improve consistency without labeled data.

## Quotes / details to potentially cite

- Method summary: majority-vote pseudo-labels (strict majority) + alignment pulling low-confidence/non-majority variants toward a high-confidence majority subset.
- Reported aggregate improvements (across 11 datasets): agreement +11.62%, mean F1 +8.94%, variance reduction 3.29%.
- Metric definition: PoP_o as probability two sampled prompt variants predict the same label (computed from vote counts).
