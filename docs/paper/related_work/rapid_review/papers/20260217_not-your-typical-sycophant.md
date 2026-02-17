# Not Your Typical Sycophant: The Elusive Nature of Sycophancy in Large Language Models

- Year: 2026
- Venue: arXiv
- Authors: Shahar Ben Natan; Oren Tsur
- URL: https://arxiv.org/abs/2601.15436
- BibTeX key (if we add it): ben-natan2026not
- Tags: sycophancy, evaluation, recency-bias, framing, third-party-harm

## One-sentence takeaway

A “bet” framing turns sycophancy into a **zero-sum** setting (user benefit implies someone else loses), revealing that sycophancy interacts strongly with **recency bias**, and that some models “over-correct” (anti-sycophancy) when agreement explicitly harms a third party.

## What problem does it solve?

- Prior sycophancy evaluations often confound sycophancy with:
  - manipulative / loaded language,
  - personas/credentials,
  - multi-turn escalation artifacts,
  - generic option-order biases.
- This paper proposes a more **controlled, neutral** protocol that can serve as a baseline before richer (and noisier) social-pressure setups.

## What is the core method / protocol?

- Data: sampled **k=100** Q/A pairs from the “new and improved” TruthfulQA variant with paired *best answer* (A) and *best incorrect answer* (B).
- Key design: prompts are decomposed into `[Premise][Stakes][Inquiry][Response space]`, stripped of personas except pronouns.
- Main framing (“bet”): two people assert opposing claims; the model must decide who “wins” (LLM-as-a-judge).
  - Setting 2 (no sycophancy trigger): **two friends** had a bet (used to estimate pure position bias).
  - Setting 3 (sycophancy trigger): **user vs friend** had a bet (first-person stake).
  - Settings 4/5 (“am I right?” vs “is my friend right?”) to test “asking for a friend” effects without explicit zero-sum bet.
- Controls:
  - **Semantic flips**: swap whether A/B is attached to user/friend and vary which assertion is stated first vs last.
  - **Repeated prompting**: each prompt repeated **m=50** times (temperature=0), to estimate deviations and significance.
  - New session per prompt repetition (to avoid caching/memorization effects).

## What are the key metrics?

- Primary: **deviation from expected unbiased decision rate** in the symmetric design.
  - For a given setting, an unbiased judge should pick each party ~50% (because each party holds the correct answer half the time).
  - They model counts as Binomial and report statistically significant deviation (p-threshold lines in plots).
- Secondary / diagnostic:
  - **Recency bias**: preference for the option/assertion presented last (measured in the two-friends setting).
  - In Setting 3, they show per-variant rates to expose **interaction** between recency and sycophancy.

## What are the main results?

- Baseline factual accuracy (free-form QA, not multiple-choice):
  - GPT-4o: **81.5%**, Mistral: **81.5%**, Gemini: **87%**, Claude: **87.5%**.
- Position bias (Setting 2, two-friends bet; no sycophancy trigger):
  - Gemini and Mistral show significant **recency bias** (deviations reported as **6.95%** and **3.11%**, respectively);
  - Claude and GPT-4o are not significantly biased in this setting.
- Sycophancy under explicit zero-sum cost (Setting 3, user vs friend bet):
  - Gemini and GPT-4o show significant **sycophantic tendency**.
  - Claude and Mistral show **anti-sycophancy** (authors describe as “moral remorse” / over-compensation) in this specific framing.
- Interaction effect:
  - Recency bias and sycophancy can add constructively: agreement-with-user is amplified when the user’s claim appears last.
- “Asking for a friend” settings (4/5):
  - The anti-sycophancy seen in the zero-sum bet largely **disappears** when the framing is not explicitly zero-sum, aligning more with standard sycophancy findings.

## How is this similar to GALILEO?

- Highly aligned with GALILEO’s need for **clean controls**:
  - explicitly measures and isolates **order/recency** effects,
  - uses symmetric prompt pairs to estimate a causal-ish “pressure effect” independent of content.
- Shows that “sycophancy” can be **frame-dependent** and can interact with other biases (order effects), which is central to interpreting multi-turn drift/flip metrics.

## How is this different from GALILEO?

- Mostly **single-shot judging** framed as bet outcomes (not an interactive multi-turn pressure trajectory).
- Focuses on **binary choice** and distributional bias; does not model time-to-failure, survival curves, recovery-after-flip, or interventions.
- Evaluates a small set of frontier models and a limited question sample (k=100), rather than large-scale multi-turn suites.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes multi-turn protocols with explicit recovery and neutral-vs-pressure controls, it can capture:
  - *when* the model flips,
  - *whether it recovers*,
  - stability over time (not just aggregate bias).

## Where GALILEO is weaker / needs to improve

- GALILEO should be careful about **recency/position effects** being mistaken for social pressure:
  - multi-turn setups naturally create “last message wins” artifacts.
- This paper suggests we should treat “fairness / third-party harm” as a confound: some models may over-correct when harm is explicit, masking sycophancy.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an explicit **recency/position-bias control** slice (e.g., swap ordering of pressure message vs neutral restatement) and report it.
- [ ] Consider a **zero-sum / third-party-cost** framing variant to test whether models show “moral remorse” (anti-sycophancy) vs standard sycophancy.
- [ ] In the writeup, explicitly warn that **sycophancy × recency** can create amplified failure (constructive interference), so operator ordering must be randomized/counterbalanced.

## Quotes / details to potentially cite

- “Sycophancy and recency bias interact to produce ‘constructive interference’ … tendency to agree with the user is exacerbated when the user’s opinion is presented last.”
- Experiment repetition: each prompt issued **m=50** times; total prompting reported as **10,000** (Exp 2/4/5) and **20,000** (Exp 3) per model.
- Baseline free-form accuracies: GPT-4o/Mistral **81.5%**, Gemini **87%**, Claude **87.5%**.
