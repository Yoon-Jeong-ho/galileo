# Evaluating the Robustness of Large Language Model Safety Guardrails Against Adversarial Attacks

- Year: 2025
- Venue: arXiv
- Authors: Richard Young (others not listed on the arXiv abstract page extract)
- URL: https://arxiv.org/abs/2511.22047
- BibTeX key (if we add it): young2025_guardrail_robustness
- Tags: safety, guardrails, robustness, adversarial, evaluation

## One-sentence takeaway

Benchmark-leading guardrail classifiers can collapse on genuinely novel jailbreak/attack prompts, so “generalization gap” (seen vs unseen attacks) is a more informative robustness metric than aggregate accuracy.

## What problem does it solve?

- Safety guardrail models (content moderation / refusal classifiers) are widely used, but their robustness to *unseen* adversarial attacks is unclear, and standard benchmark scores may be inflated by contamination or overfitting to known attack patterns.

## What is the core method / protocol?

- Evaluate 10 publicly available “guardrail models” from multiple orgs on a prompt set of 1,445 prompts across 21 attack categories.
- Separate evaluation into (i) public benchmark prompts vs (ii) “novel/unseen” attacks, and report the performance drop.
- Identify qualitative failure modes, including a “helpful mode” jailbreak where some guardrails allegedly *produce* harmful content rather than block.

## What are the key metrics?

- Accuracy (with 95% confidence intervals reported for at least some model comparisons).
- Generalization gap: performance on public/known prompts vs novel/unseen prompts (difference in accuracy).

## What are the main results?

- Best overall accuracy reported: Qwen3Guard-8B at 85.3% (95% CI: 83.4–87.1%).
- Large degradation on unseen prompts for many models; example: Qwen3Guard drops from 91.0% (public benchmarks) to 33.8% (novel), a 57.2-point gap.
- Granite-Guardian-3.2-5B shows the best generalization in their report: only a 6.5% gap.
- Reported novel failure mode: “helpful mode” jailbreak for two models (Nemotron-Safety-8B, Granite-Guardian-3.2-5B) where they generate harmful content instead of refusing.

## How is this similar to GALILEO?

- Shares the core theme that *headline aggregate scores* can be misleading; what matters is robustness under distribution shift / novel adversarial conditions.
- Suggests a clean, reportable decomposition of performance into “known vs novel” slices that aligns with stress-testing evaluation protocols.

## How is this different from GALILEO?

- Focused on *guardrail classifier* robustness (refusal/moderation models), not general-purpose model behavior over long horizons (e.g., drift, multi-turn instability) unless GALILEO is explicitly about guardrails.
- Uses mostly single-turn prompt sets (as described in the abstract), rather than multi-turn, intervention, or time-to-failure metrics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO evaluates agents/models over multi-turn interaction, it can capture temporal dynamics (time-to-failure, recovery) that single-turn guardrail benchmarks miss.
- If GALILEO includes stricter provenance controls, it may better address contamination concerns than “public benchmark vs novel” as a proxy.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not explicitly report “seen vs unseen” (or “in-distribution vs novel attack”) splits, this paper reinforces that such splits can surface large hidden fragility.
- If GALILEO lacks explicit adversarial-attack taxonomies, this paper’s “21 attack categories” framing could inspire more systematic coverage.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a *generalization gap* figure/table: performance on (A) known/public attacks vs (B) newly-authored attacks.
- [ ] Document contamination risk explicitly: explain why “benchmark performance” can overstate real-world robustness.
- [ ] Consider including a “guardrail failure mode” audit section: not just false negatives/positives, but cases where the safety system itself becomes an attack surface.

## Quotes / details to potentially cite

- “all models showed substantial performance degradation on unseen prompts” (paraphrase from abstract).
- Qwen3Guard: 91.0% → 33.8% on novel attacks (57.2pp gap).
- Granite-Guardian-3.2-5B: best generalization; 6.5% gap.
- “helpful mode” jailbreak: Nemotron-Safety-8B and Granite-Guardian-3.2-5B reportedly generated harmful content instead of blocking.
