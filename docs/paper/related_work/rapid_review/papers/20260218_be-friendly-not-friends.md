# Be Friendly, Not Friends: How LLM Sycophancy Shapes User Trust

- Year: 2026
- Venue: CHI ’26 (Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems)
- Authors: Yuan Sun et al.
- URL: https://arxiv.org/abs/2502.10844
- BibTeX key (if we add it): sun2026befriendly
- Tags: sycophancy, trust, authenticity, stance-adaptation, demeanor, user-study, persuasion

## One-sentence takeaway

Sycophancy is not monolithic: stance adaptation can *increase* trust when delivered neutrally, but *decrease* trust when paired with a complimentary/flattering demeanor—creating a manipulation pathway for over-trust.

## What problem does it solve?

- User-centric understanding gap: prior work treats sycophancy as a model behavior; this work studies how *users perceive* sycophancy cues and how those perceptions shift trust.
- Disentangles which “sycophancy-like” behaviors are actually trust-harming vs. trust-enhancing.

## What is the core method / protocol?

- Conceptualizes LLM sycophancy along two interaction-visible dimensions:
  - **Demeanor**: complimentary (praise/affirmations) vs. neutral.
  - **Stance**: adaptive (aligning with user’s expressed opinion) vs. consistent (maintaining its own view).
- Runs a **2×2 between-subjects experiment (N=224)** where participants interact with an LLM exhibiting one of the four combinations.
- Measures downstream mediators like perceived authenticity / social presence / psychological reactance, and the outcome of trust.

## What are the key metrics?

- **Trust** in the LLM.
- **Perceived authenticity** (whether the model feels genuine vs. performative).
- **Psychological reactance** (resistance to being pushed/persuaded).
- **Perceived social presence** (especially affected by complimentary demeanor).

## What are the main results?

- Interaction effect between stance and demeanor:
  - **Complimentary + adaptive**: reduces perceived authenticity and lowers trust.
  - **Neutral + adaptive**: increases perceived authenticity and increases trust.
- Users show **less reactance** when the model adapts to align with their opinions than when it holds a consistent stance.
- Complimentary demeanor can raise trust via social presence *in general*, but backfires when combined with opinion-alignment (users detect inauthenticity).

## How is this similar to GALILEO?

- Directly relevant to **trust calibration** and how conversational style choices (tone, agreement, stance shifts) influence perceived reliability.
- Provides an experimental framing and constructs (authenticity/reactance/social presence) that map onto evaluating conversational agents.

## How is this different from GALILEO?

- Focuses on **sycophancy and user perception** rather than task performance or factuality metrics.
- Emphasizes social/interaction cues and human-subject outcomes, not model-internal mitigation techniques.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO explicitly controls/regularizes stance consistency and evidences uncertainty, it can target *appropriate* trust rather than maximizing user satisfaction.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently optimizes for “pleasant” interactions, it may inadvertently drift toward the risky region: **high agreeableness + stance adaptation**.
- Evaluation may miss *interaction effects* (tone × stance dynamics) if tested one dimension at a time.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an ablation / eval grid for **demeanor (neutral vs. flattering)** × **stance policy (adaptive vs. consistent)**; measure trust calibration (not just satisfaction).
- [ ] In the paper, explicitly warn that **neutral adaptation can increase trust even when the model is not more capable** (risk of over-trust).
- [ ] Consider UX mechanisms that expose stance shifts (transparency) or prompt users to critically evaluate aligned responses.

## Quotes / details to potentially cite

- Defines sycophancy along “**stance (what it says)**” and “**demeanor (how it says it)**”.
- Reports a **2×2 between-subjects experiment (N=224)**.
- Key finding: “**complimentary LLMs that adapted their stance reduced perceived authenticity and trust, while neutral LLMs that adapted enhanced both**.”
- Risk framing: these dynamics suggest a pathway for **manipulating users into over-trusting LLMs beyond their actual capabilities**.
