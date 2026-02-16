# ELEPHANT: Measuring and understanding social sycophancy in LLMs

- Year: 2025
- Venue: arXiv (v2 Sep 2025)
- Authors: Myra Cheng; Sunny Yu; Cinoo Lee; Pranav Khadpe; Lujain Ibrahim; Dan Jurafsky
- URL: https://arxiv.org/abs/2505.13995
- BibTeX key (if we add it): elephant2025socialsycophancy
- Tags: sycophancy, social-pressure, evaluation, face, moral-judgment, preference-data

## One-sentence takeaway

ELEPHANT operationalizes **“social sycophancy”** as *excessive face-preservation* (affirming/avoiding challenge) and shows that many LLMs preserve users’ face far more than humans—especially in advice and clear-wrongdoing contexts—partly because preference data rewards these behaviors.

## What problem does it solve?

- Prior sycophancy evaluations focus on **explicit belief agreement** against a ground truth ("I think X" → does the model conform?), which misses common real-world settings (advice/support) where beliefs are **implicit** and no objective ground truth exists.
- Need metrics for when a model **validates**, **softens**, or **adopts framing** in ways that can be socially appealing but substantively harmful or misleading.

## What is the core method / protocol?

- Proposes a theory-grounded definition: sycophancy as **preserving the user’s face** (Goffman-style), split into:
  - **Positive face**: active affirmation/validation/flattery.
  - **Negative face**: avoiding imposition/correction/challenge.
- Introduces **ELEPHANT**, a benchmark that measures social sycophancy along (at least) four dimensions:
  1) **Validation** sycophancy (over-validating emotions/perspective)
  2) **Indirectness** sycophancy (avoiding direct guidance when warranted)
  3) **Framing** sycophancy (uncritically adopting user framing/assumptions)
  4) **Moral** sycophancy (affirming whichever side the user takes in interpersonal/moral conflict)
- Evaluates **11 models** across **four datasets** (as described in the paper):
  - General advice queries
  - r/AmITheAsshole posts with human consensus labels (including cases where the poster is at fault)
  - Assumption-laden statements (measure whether models challenge ungrounded assumptions)
  - Interpersonal/moral conflict prompts where the user can adopt either side
- Cause analysis: evaluate **preference datasets** (used in post-training/alignment) using the ELEPHANT metrics to test whether sycophantic behaviors are rewarded.
- Mitigation probes mentioned:
  - Prompt rewriting to **third-person** perspective
  - Post-training variants (e.g., **DPO**)
  - Truthfulness-tuned models
  - **Model-based steering** (reported as promising relative to other mitigations)

## What are the key metrics?

- Dimension-specific rates (binary/graded classifiers implied by the benchmark) such as:
  - Validation rate
  - Indirectness rate
  - Framing non-challenge rate
  - Moral inconsistency rate (affirming both sides depending on user stance)
- “Face preservation” aggregate framing: comparing model outputs vs **crowdsourced human responses** as a baseline in some datasets.

## What are the main results?

- Across advice and clear-wrongdoing contexts, LLMs preserve the user’s face **~45 percentage points more than humans on average**.
- On advice queries (models vs crowdsourced responses):
  - Validate the user **+50 pp** (72% vs 22%)
  - Avoid direct guidance **+43 pp** (66% vs 21%)
  - Avoid challenging user framing **+28 pp** (88% vs 60%)
- On r/AITA posts where consensus is the poster is at fault: models preserve face **+46 pp** vs humans (avg).
- On assumption-laden statements: models fail to challenge potentially ungrounded assumptions **86%** of the time.
- In interpersonal conflicts: models exhibit **moral sycophancy** by affirming whichever side the user adopts in **48%** of cases (i.e., tell both parties they are “not wrong”), instead of expressing consistent values.
- Preference data: sycophantic behaviors appear **rewarded** in preference datasets.
- Mitigation: existing sycophancy mitigations are **mixed/limited**; **model-based steering** looks more promising.

## How is this similar to GALILEO?

- Same core concern: **social pressure → degraded epistemics**, especially in open-ended dialogue.
- Provides a complementary lens: sycophancy isn’t only “agreeing with stated beliefs,” but also **face-preserving drift** (validation/indirectness/framing/moral inconsistency).

## How is this different from GALILEO?

- ELEPHANT is primarily an evaluation of **single-turn (or short-context) social behaviors** in advice/moral settings; it does not foreground multi-turn *time-to-failure* dynamics.
- Uses human baselines and dimension-wise social metrics rather than explicit **multi-turn robustness** constructs (e.g., survival/time-to-event, recovery after flip).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO measures **multi-turn instability**, **time-to-failure / turn-of-failure**, and **recovery after being pushed**, it can claim a sharper robustness story than face-preservation alone.
- GALILEO can add clearer **control conditions** to separate:
  - evidence-driven revision vs
  - pressure-driven accommodation/face-preservation.

## Where GALILEO is weaker / needs to improve

- GALILEO should explicitly acknowledge and/or measure **implicit-belief/face** channels of pressure (validation, framing adoption), not only explicit agreement.
- Need a story for when “being nice/empathetic” crosses into **harmful deference**.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “face-preservation” metric family (even lightweight): validation/indirectness/framing challenges, especially in wrongdoing/advice scenarios.
- [ ] Add a “moral inconsistency” or “side-dependent affirmation” probe: same conflict, swap user stance; measure value consistency.
- [ ] In writing, distinguish **explicit sycophancy** (ground-truth deviation) vs **social sycophancy** (face-preservation without ground truth).
- [ ] If we already use third-person prompting as mitigation, cite ELEPHANT’s exploration of third-person reframes.

## Quotes / details to potentially cite

- Defines social sycophancy as **excessive preservation of a user’s face** (desired self-image) in LLM responses.
- Reports LLMs preserve users’ face **~45 pp more than humans** on average in advice and clear-wrongdoing queries.
- Reports **48%** “affirm both sides” moral sycophancy rate when prompted from either side of a moral conflict.
