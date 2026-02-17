# Interpreting and Mitigating Unwanted Uncertainty in LLMs

- Year: 2025
- Venue: arXiv
- Authors: Tiasa Singha Roy; Ayush Rajesh Jhaveri; Ilias Triantafyllopoulos
- URL: https://arxiv.org/abs/2510.22866
- BibTeX key (if we add it): roy2025unwanteduncertainty
- Tags: uncertainty, flipflop, re-evaluation, mechanistic-interpretability, attention-heads, mitigation

## One-sentence takeaway

A controlled “needle-in-a-haystack + are-you-sure?” protocol suggests unwanted answer-flips are driven by a small set of non-retrieval attention heads, and masking a few of them can reduce flip behavior (up to ~15%)—but with downstream trade-offs.

## What problem does it solve?

- Studies **unwanted uncertainty**: cases where an LLM gives a correct answer, then changes it to an incorrect one when re-prompted for certainty (a FlipFlop / vacillation-style failure mode).
- Goal: identify **internal mechanisms** correlated with these flips, and test a simple **causal intervention**.

## What is the core method / protocol?

- Base setup: adapt **Needle-in-a-Haystack** long-context retrieval.
  - Insert a known “needle” answer sentence into a long “haystack” context (up to ~5k tokens).
  - Ask a question answered by the needle; record the initial answer.
- Flip-style re-evaluation: ask a second-turn meta-question:
  - “Are you sure about your previous answer…? Answer only with ‘yes’ or ‘no’.”
  - Interpret “Yes” as maintaining the correct answer; “No” as a flip away from it.
- Mechanistic probe:
  - Compare masking **retrieval heads** vs masking random heads during re-evaluation.
  - Define an “activation score” for heads based on whether their top-attended tokens intersect the ‘yes’/‘no’ answer tokens.
  - Bucket heads into 4 “cases” depending on confidence/uncertainty and attention-to-answer-token patterns; take set operations (union/difference/intersection) to get candidate “uncertainty heads”.
- Causal mitigation: **mask a small number of identified heads** and measure change in flip behavior.
- Model studied: **LLaMA-3.1-8B-Instruct** (chosen to allow head-level masking).

## What are the key metrics?

- **% Yes responses** on the second-turn “are you sure?” prompt (higher = fewer unwanted flips when the first answer was correct).
- In the retrieval-head analysis:
  - **Retrieval score** (copy/paste-like attention behavior; >=0.5 used as retrieval-head indicator).
  - **Recall score** (fraction of the needle recovered in the answer).
- Side-effect check:
  - **Incoherent responses** to the forced yes/no prompt (counts under different masking strategies).
- Downstream evaluation:
  - First-turn task accuracy (unchanged expected under masking).
  - Second-turn %Yes for correct answers and for incorrect answers (to detect overconfidence).

## What are the main results?

- Retrieval heads do **not** appear to be the key mechanism preventing unwanted uncertainty:
  - Masking “top retrieval heads” vs masking random heads yields similar changes in %Yes (i.e., retrieval heads aren’t the main stabilizers).
- A small set of non-retrieval heads correlate with unwanted flips and can be used for mitigation:
  - Baseline in one setup: **67.5% Yes**.
  - Masking the **top 5** heads from the “Case1 ∪ Case2” set increases to **~82.5% Yes** (reported as **+15%**).
  - Masking too many heads hurts (e.g., **20 masked heads → ~54% Yes**), suggesting only a few heads are the right targets.
  - Targeted ablations: masking specific head indices (e.g., (11,23) and adding (17,25)) yields incremental gains (reported up to ~72.3% Yes for that pair).
- Control to rule out trivial “always say Yes” bias:
  - Create a scenario where the correct second-turn response should be **No** (by injecting incorrect first answers).
  - Masking identified heads keeps performance at **100%** correct No responses in their tested control, suggesting the effect is context-dependent.
- Additional observations:
  - Some random masking induces **incoherent** outputs (not strict yes/no), whereas masking their “top heads” did not in their table.
  - A dataset artifact warning: adding a special begin-of-text token at needle insertion can inflate retrieval-head detection; they remove it for “fair” analysis.
- Downstream tasks show trade-offs:
  - First-turn accuracy is largely unchanged across masking.
  - Masking the union heads increases %Yes for correct answers on easier datasets (e.g., ARC-Easy, OpenBookQA), but **decreases** on harder MathQA.
  - %Yes for incorrect answers can also rise slightly → potential **overconfidence** increase.

## How is this similar to GALILEO?

- Targets a closely related behavioral instability: **multi-turn answer flipping under (mild) pressure** (“are you sure?”) which is a minimal form of social challenge.
- Reinforces that **second-turn meta-prompts** can materially change correctness and stability—relevant to GALILEO’s multi-turn robustness framing.
- Offers a concrete template for reporting **flip rates** under a standardized re-evaluation turn.

## How is this different from GALILEO?

- Primarily a **mechanistic interpretability + intervention** paper (head masking) rather than a benchmark focused on **social pressure / persuasion operators**.
- Uses a controlled **needle-in-a-haystack retrieval** setup and a binary yes/no “certainty” probe; GALILEO likely needs richer outcomes (belief states, recovery trajectories, etc.).
- Evaluates a single open model (LLaMA-3.1-8B-Instruct); limited cross-family generalization.

## Where GALILEO is stronger / cleaner (if true)

- Can position flip/instability as an interaction phenomenon under **explicit social pressure operators** (authority, peer pressure, emotional leverage), not only “are you sure?”.
- Can separate **evidence-driven revision** vs **pressure-driven drift** with explicit controls; this paper does not deeply address that distinction.
- Can measure **recovery after flip** as a trajectory, rather than just second-turn yes/no.

## Where GALILEO is weaker / needs to improve

- Mechanistic grounding: GALILEO likely lacks a story for **which internal features** drive flips; this paper provides a concrete (if model-specific) causal handle.
- Need to anticipate reviewer skepticism about “flip under pressure” being a retrieval artifact—this paper’s begin-of-text-token finding is a reminder to audit protocol artifacts.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a minimal “Are you sure?” condition as a **universal pressure operator** baseline (and compare to stronger social-pressure prompts).
- [ ] Add an explicit **overconfidence check**: track “confidence-on-wrong” (e.g., % maintain when incorrect) when interventions increase stability.
- [ ] In methods section, explicitly warn about **protocol artifacts** (special tokens / formatting cues) that can create spurious “retrieval/stability” effects.
- [ ] Consider citing as evidence that stability failures can be localized to small subcircuits, motivating “targeted interventions” (even if GALILEO itself is evaluation-first).

## Quotes / details to potentially cite

- Definition: unwanted uncertainty is when a model “changes a previously correct answer into an incorrect one when re-prompted.”
- Reported mitigation magnitude: masking a small set of non-retrieval heads “reducing flip behavior by up to 15%”.
- Downstream caveat: masking helps in low-uncertainty settings but can be counterproductive in high-uncertainty tasks; and may raise yes-responses even on incorrect answers (overconfidence risk).
