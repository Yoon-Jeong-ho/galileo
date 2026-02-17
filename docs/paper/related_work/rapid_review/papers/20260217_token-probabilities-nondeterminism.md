# Beyond Reproducibility: Token Probabilities Expose Large Language Model Nondeterminism

- Year: 2026
- Venue: arXiv
- Authors: Tairan Fu; Gonzalo Martínez; Javier Conde; Carlos Arriaga; Pedro Reviriego; Xiuyuan Qi; Shanshan Liu
- URL: https://arxiv.org/abs/2601.06118
- BibTeX key (if we add it): fu2026beyondreproducibility
- Tags: nondeterminism, reproducibility, token-probabilities, inference, evaluation

## One-sentence takeaway

Even when generation is configured “deterministically” (e.g., temperature=0), GPU execution can induce meaningful *token-probability* variation—largest in mid-probability regions (≈0.1–0.9)—suggesting nondeterminism can be quantified without repeatedly sampling full outputs.

## What problem does it solve?

- Prior work on LLM nondeterminism largely measures **output-text differences** across runs, which is a coarse lens (and can miss probability-level drift that doesn’t flip the argmax).
- The paper aims to **characterize and quantify nondeterminism at the token-probability level**, and to understand when it is likely to matter for downstream generation/evaluation.

## What is the core method / protocol?

- Treat nondeterminism as **uncertainty/variation in next-token probabilities** across repeated inferences on the same prompt under “deterministic” settings.
- Compute variation statistics across runs for token probabilities (the paper proposes two probability-variation metrics; the main idea is measuring dispersion/range across runs at each step / token-probability value).
- Compare trends across **models, configurations, and hardware**, including (notably) reporting results on **Huawei GPUs**.

## What are the key metrics?

- Token-level probability variation metrics (e.g., dispersion such as standard deviation and/or range of probabilities for the same token position across repeated runs).
- Analysis stratified by the **magnitude of the probability** (finding a characteristic “mid-probability band” where nondeterminism is largest).

## What are the main results?

- Across evaluated models, probability-variation curves show **similar trends and similar absolute magnitudes**.
- Nondeterminism effects are **significant when token probabilities are ~0.1–0.9** and much smaller when probabilities are near **0 or 1**.
- Implication argued: when **temperature > 0**, these probability perturbations are likely to cause **non-negligible output changes** (since sampling amplifies mid-probability fluctuations).
- The authors suggest a practical shortcut: potentially **estimate nondeterminism impact from a single inference** by inspecting token probabilities, rather than repeatedly running the same prompt many times.
- They release a public dataset of their evaluation results.

## How is this similar to GALILEO?

- GALILEO’s multi-turn robustness metrics (e.g., survival/time-to-failure, drift/instability curves) can be **confounded by inference nondeterminism**, especially when failures occur near decision boundaries.
- The “mid-probability band is most unstable” result is conceptually aligned with **boundary sensitivity**: models that operate with more ambiguous next-token distributions may show more instability.

## How is this different from GALILEO?

- This work is about **hardware/execution nondeterminism** (floating-point / kernel scheduling / batching effects) and analyzes **token probabilities**, not multi-turn conversational robustness under adversarial interaction.
- No explicit multi-turn protocol, persona pressure, or evaluator/judge setup.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO controls for decoding settings and reports across seeds/replicates, it can connect instability to **interaction protocols** (not just execution artifacts).
- GALILEO can show how nondeterminism manifests in **task-level failures over turns**, which is more directly relevant to safety/robustness claims.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently attributes variability mainly to prompting/interaction factors, it may under-account for **inference nondeterminism** as a baseline noise source.
- If GALILEO compares models/providers, hardware-level nondeterminism differences could masquerade as model differences unless measured/controlled.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short “threats to validity” paragraph: **GPU nondeterminism can shift token probabilities even at temperature=0**, affecting multi-turn stability metrics.
- [ ] Consider reporting a lightweight nondeterminism diagnostic: for a small prompt set, log **token probability dispersion** across repeated runs (or at least output variance) to bound noise.
- [ ] When interpreting cross-model differences, note the paper’s claim that **probability-level nondeterminism magnitudes are similar across models**, so task-level variance may come from (i) where probabilities sit (near 0/1 vs mid-band) and/or (ii) response length.

## Quotes / details to potentially cite

- “This work takes a closer look at nondeterminism by analyzing the variations on the token probabilities, not on the generated text.”
- “The effects of nondeterminism are significant for token probabilities that are in the range of 0.1 to 0.9, while they are much smaller when the probabilities are close to 0 or 1.”
- “We may be able to estimate the impact of nondeterminism by running a single inference and analyzing the token level probabilities, instead of having to run the same inference many times.”
