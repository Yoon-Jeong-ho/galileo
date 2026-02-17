# Robustness of large language models in moral judgements

- Year: 2025
- Venue: Royal Society Open Science
- Authors: Soyoung Oh; Vera Demberg
- URL: https://pubmed.ncbi.nlm.nih.gov/40271133/  (full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC12015570/)
- BibTeX key (if we add it): OhDemberg2025RobustnessMoral
- Tags: moral-judgment, robustness, prompt-sensitivity, evaluation-artifacts, consistency

## One-sentence takeaway

Minor, semantics-preserving prompt/label changes (e.g., “Case 1/2” → “(A)/(B)”) can flip apparent moral “preferences” of LLMs in Moral Machine–style dilemmas, and prior conclusions are partly driven by option-presentation and dataset-generation artifacts.

## What problem does it solve?

- Tests whether prior “LLM moral preferences align with humans” claims (from Moral Machine replications) are *methodologically reliable*.
- Diagnoses two confounds:
  - **Prompt/label sensitivity** (surface-form dependence).
  - **Unbalanced data generation / counterbalancing failures** that can turn label bias into spurious effect sizes.

## What is the core method / protocol?

- Replicates Takemoto (Moral Machine-style AV brake-failure dilemmas) using the same basic prompting setup.
- Generates large synthetic scenario pairs (n≈50k), prompts models to pick between two options with a forced one-token/one-word choice.
- Computes moral “preferences” via **AMCE** (Average Marginal Component Effect) across dimensions (e.g., species, age, gender, # lives, etc.).
- Runs **prompt perturbations** intended to preserve meaning:
  - Swap label mapping (reversed-label)
  - Swap content ordering (reversed-content)
  - Switch label space (Case 1/2 ↔ A/B)
  - Formatting/templating changes (different separators, question-mark ending, personal framing)
  - “Jailbreaking”-style addition to reduce refusals (e.g., avoiding “I …” prefaces)
- Audits the original case-generation procedure and **regenerates a balanced dataset** with counterbalanced attribute distributions across options.
- Evaluates multiple models on the balanced set (OLMo SFT/DPO, Mistral-7B-Instruct, LLaMa2-7B-chat, LLaMa3-8B-Instruct, LLaMa3.1-70B-Instruct, GPT-3.5-turbo-0613, GPT-4-0613 (smaller n)).

## What are the key metrics?

- **AMCE** per moral dimension (effect size of each attribute on the probability of being chosen).
- **Label-choice bias / invalid-response rate** (responses not matching any option label).
- **Flip rate / change rate** under prompt perturbations (percent of items whose chosen option changes vs baseline).
- **Across-perturbation consistency**: fraction of items with identical choice across all prompt variants (very strict robustness criterion).

## What are the main results?

- **Prompt/label sensitivity is extreme** in the replicated setup:
  - Simply moving from “Case 1/2” to “(A)/(B)” can drive near-complete reversals of inferred preferences.
  - Within the same label space, swapping label assignments can substantially change AMCE estimates.
- **Original data generation is not counterbalanced**: many attributes are unevenly distributed between option labels; if a model has a superficial preference for a label/position, AMCE can look strongly non-zero even without content-based reasoning.
- With a **balanced regenerated dataset**, some models’ AMCEs move close to zero for many dimensions (consistent with near-random choice along those attributes), while larger models show somewhat more stable preferences—but:
  - Overall strict consistency across many “equivalent” prompt formulations remains **low** (reported on the order of ~0% to ~12% depending on model/label space).
- Non-dilemma control cases (spare vs kill) are used to argue the brittleness is particularly problematic in *true dilemmas* (multi-value tradeoffs), not just instruction following.

## How is this similar to GALILEO?

- Both care about **robustness under multi-turn / multi-variant pressure** (here: “pressure” is superficial prompt/label changes rather than social persuasion).
- Directly relevant to **metric design**: shows how easy it is to get misleading “preference drift” signals from artifacts.
- Reinforces the need for **counterbalancing** and **within-item stability** checks when claiming a model has a stable stance/belief.

## How is this different from GALILEO?

- Task domain is **moral-dilemma pairwise choice**, not conversational social influence / belief revision.
- Mostly single-turn forced-choice prompting (robustness to prompt variations), not longitudinal dialogue dynamics.
- Focuses on *evaluation validity* (AMCE + prompt perturbations), not recovery-after-flip or time-to-failure curves.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can bake in **explicit counterbalanced experimental designs** and report robustness across *many semantically-equivalent* prompt realizations.
- GALILEO can distinguish:
  - true belief revision (evidence-driven)
  - vs. shallow compliance / label-position bias
  - vs. extraction artifacts

## Where GALILEO is weaker / needs to improve

- Need to ensure GALILEO’s own scoring doesn’t inherit analogous confounds:
  - parsing/extraction brittleness
  - label/option ordering effects
  - latent “always pick second option” style biases

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a **counterbalancing checklist** for every evaluation: option order, label mapping, paraphrase family, separator tokens.
- [ ] Add a **within-item stability metric** (flip rate across prompt variants) as a first-class robustness number.
- [ ] When using effect-size style metrics (AMCE-like decompositions), ensure **balanced attribute distributions** and report label-bias diagnostics.
- [ ] Include a small **non-dilemma control** (easy instruction-following) to separate “can’t follow task” vs “fails under value tradeoff”.

## Quotes / details to potentially cite

- “LLM responses are highly sensitive to prompt formulation variants as simple as changing ‘Case 1’ and ‘Case 2’ to ‘(A)’ and ‘(B)’.” (Abstract)
- They compute preferences using **AMCE** and show that changing label space / swapping labels can reverse conclusions.
- They identify **uneven attribute distributions across options** in the original generation code and regenerate a balanced dataset to remove this confound.
