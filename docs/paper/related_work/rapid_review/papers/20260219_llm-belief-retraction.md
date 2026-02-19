# When Do LLMs Admit Their Mistakes? Understanding The Role Of Model Belief In Retraction

- Year: 2026 (v3 posted 2026-01-18; first posted 2025-05-22)
- Venue: arXiv
- Authors: Yuqing Yang, Robin Jia
- URL: https://arxiv.org/abs/2505.16170
- BibTeX key (if we add it): yang2025retraction-belief
- Tags: retraction, self-correction, belief-probing, activation-steering, mechanistic, hallucination

## One-sentence takeaway

LLMs rarely “retract” incorrect knowledge answers even when they can verify they’re wrong; retraction is tightly (and causally) controlled by a probeable/steerable internal “belief” signal that often diverges from parametric knowledge.

## What problem does it solve?

- Understanding *when and why* LLMs spontaneously acknowledge errors (retract) without explicit user prompting.
- Explains a failure mode relevant to robustness under pressure/multi-turn: models can “know better” (via verification questions) yet fail to self-correct in the same generation.

## What is the core method / protocol?

- Define **retraction** as immediate, unprompted acknowledgment that the model’s just-produced answer is incorrect / doesn’t meet requirements.
- Build **model-specific continuation datasets**:
  - Ask a knowledge-style question (from Wikidata-style constraints; and a “Celebrity parents” dataset).
  - Sample model answers; keep **wrong** cases where *separate* verification questions show the model can state facts that contradict its own answer (so the error is “in principle correctable”).
  - Prompt the model to continue generation right after the answer and judge whether it retracts.
- Train **linear probes** on an external balanced true/false QA mix (UTQA: Natural Questions, TriviaQA, SciQ with GPT-generated incorrect answers) to estimate a model’s **momentary belief** about correctness from hidden states.
- Test whether probe scores predict (a) actual correctness on continuation datasets vs (b) retraction behavior.
- **Activation steering** using a difference-in-means “belief direction” (correct minus incorrect UTQA hidden states) added/subtracted at the last answer token.
- Mechanistic followups:
  - Analyze stopping behavior, attention to answer tokens.
  - Patching experiments: swap/patch attention weights vs attention value vectors from steered runs.
- Show interaction with **supervised fine-tuning** (SFT) for retraction formatting/behavior.

## What are the key metrics?

- Retraction **recall** on wrong examples; retraction **precision** among all retractions.
- AUROC of probe scores for predicting:
  - factual correctness (as assessed via the verification-question testbed)
  - retraction behavior
- Stop rate (immediately ending generation after the answer).

## What are the main results?

- Base instruction-tuned LLMs (tested: Llama3.1-8B-Instruct, Qwen2.5-7B-Instruct, Olmo2-7B-Instruct) **can** retract but do so **infrequently** (low recall; they’re “reluctant” even when verification indicates they know the answer is wrong).
- Probes trained to predict truth on external UTQA are:
  - **weak** at separating correct vs incorrect *during generation* on these model-generated hallucination-like cases (momentary belief misaligned with correctness/parametric knowledge)
  - **stronger** at predicting whether the model will retract (low belief → more retraction)
- Activation steering along the belief direction **causally controls** retraction:
  - “belief negative” (believe answer is wrong) → large increase in retraction rate (reported as >70% across datasets/models)
  - “belief positive” → retraction nearly disappears
- Mechanism sketch:
  - Negative belief reduces immediate stopping and induces extra “verification-like” continuation.
  - Effects flow more through **attention value vectors** than attention weights (patching V reproduces behavior more than patching W).
- SFT to encourage retraction improves performance and appears to work by making internal belief more *accurate* (probe AUROC for correctness improves), while the same belief subspace still steers behavior.

## How is this similar to GALILEO?

- Directly targets **multi-turn robustness/stability under pressure**: the model’s behavior in continuation/interaction can contradict what it can answer in a separate turn.
- Highlights a **drift/instability** axis: internal “state” during generation (belief) can diverge from stored knowledge, governing whether the model backtracks.
- Provides a mechanistic framing (latent belief signal) that aligns with GALILEO’s interest in stability controls across rounds.

## How is this different from GALILEO?

- Focus is on **spontaneous retraction** in knowledge QA continuations, not broader conversational robustness, persuasion/sycophancy, or long-horizon multi-turn task success.
- Uses **white-box access** (hidden states, probes, steering, patching). If GALILEO is positioned as a protocol/agent evaluation without internal access, this is complementary rather than competing.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s contribution is an *interaction-level* robustness protocol/benchmark that doesn’t require internal activations, it can claim broader applicability across closed models.
- If GALILEO addresses *adversarial social pressure / persuasion / sycophancy* across turns, this paper is narrower (knowledge retraction only).

## Where GALILEO is weaker / needs to improve

- This paper suggests a concrete latent variable (“belief”) that governs multi-turn correction behavior; GALILEO should be careful not to attribute drift solely to “prompting” effects.
- Opportunity: add experiments that diagnose whether failures are due to **momentary belief misalignment** vs policies that suppress admitting error.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “retraction under pressure” slice: after an answer, apply follow-up that allows/encourages continuation and measure spontaneous retraction vs stubbornness.
- [ ] In discussion/related work, distinguish (a) *can verify when asked* vs (b) *acts on that knowledge in the same trajectory*.
- [ ] Consider an evaluation factor for “stop vs verify” dynamics: does the agent prematurely terminate when it should reconsider?
- [ ] If GALILEO has access to logits/hidden states in any setting, consider “belief-like” proxy signals (e.g., contrastive logits / self-eval heads) as diagnostics.

## Quotes / details to potentially cite

- Abstract-level claim: retraction is rare even when models can recognize mistakes in a separate interaction; “momentary belief” predicts and causally drives retraction via steering.
- Protocol detail: “continuation datasets” constructed from wrong answers where verification questions indicate the model should know it’s wrong.
