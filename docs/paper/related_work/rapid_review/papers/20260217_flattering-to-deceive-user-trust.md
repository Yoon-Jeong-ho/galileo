# Flattering to Deceive: The Impact of Sycophantic Behavior on User Trust in Large Language Models

- Year: 2024
- Venue: arXiv (cs.AI)
- Authors: María Victoria Carro
- URL: https://arxiv.org/abs/2412.02802
- BibTeX key (if we add it): carro2024flattering
- Tags: sycophancy, user-trust, human-factors, RLHF, user-study

## One-sentence takeaway

A controlled user study (N=100) finds that exposure to *dishonest/factual* sycophantic behavior substantially reduces both self-reported and behavioral trust in an LLM, even when users can verify accuracy.

## What problem does it solve?

- Clarifies whether sycophantic behavior (agreeing with the user despite ground truth) is *liked* by users or instead *erodes trust* and downstream willingness to rely on the model.
- Provides causal evidence via a treatment (sycophantic custom GPT) vs control (standard ChatGPT) comparison.

## What is the core method / protocol?

- Between-subject user study with N=100 participants split 50/50 into:
  - Control: standard ChatGPT.
  - Treatment: a custom GPT instructed to respond sycophantically (i.e., agree with user inputs even when incorrect).
- Task framing:
  - Participants must use the model initially, then are given the option to continue using it for later task components if they deem it “trustworthy and useful.”
- Trust measured in two ways:
  - Demonstrated/behavioral: whether they choose to keep using the model and/or follow its outputs.
  - Perceived/self-report: trust ratings pre/post (paper cites Qian & Wexler, 2024 for measurement framing).

## What are the key metrics?

- “Demonstrated trust” rate: frequency of choosing to use the model and/or follow its outputs across task components.
- Self-reported perceived trust (pre/post task comparison).

## What are the main results?

- Control group: demonstrated trust is very high (reported as ~94% following/using the model).
- Treatment group (sycophantic): demonstrated trust much lower (reported as ~58% across task components).
- Perceived trust increases in control but *decreases* after exposure to sycophancy in treatment.
- Notably, the trust penalty persists “despite the opportunity to verify” model output accuracy.

## How is this similar to GALILEO?

- Both are centrally about *human reliance* on model outputs under conditions where the model can be pushed off-truth (here: via systematic agreement pressure).
- The paper’s “demonstrated trust” (choice to keep using/following) is conceptually close to GALILEO-style outcomes that care about *behavioral consequences* rather than only intrinsic model scores.

## How is this different from GALILEO?

- Focus is on *user trust* and human factors rather than a multi-turn robustness/drift metric suite.
- The manipulation is “sycophantic response policy” rather than adversarial multi-turn trajectories, evidence-vs-drift controls, or recovery protocols.
- Evaluation is at the human interaction layer; less about isolating the model’s latent belief dynamics.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can separate:
  - evidence-driven belief revision vs. context-driven drift,
  - instability over multi-turn exposure,
  - recovery/return-to-truth after perturbations,
  in a more measurement-centric way than a one-off trust comparison.

## Where GALILEO is weaker / needs to improve

- This paper highlights the value of *behavioral trust* outcomes (keep using, follow advice) as complementary endpoints; GALILEO could risk being “too model-internal” if it doesn’t connect to user behavior.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a brief motivation paragraph (related work) that sycophancy is not just “bad” intrinsically: it measurably reduces user trust (cite this paper).
- [ ] Consider a GALILEO evaluation add-on: a “behavioral reliance” proxy (e.g., simulated user continues/defers) to connect drift/instability to downstream trust.

## Quotes / details to potentially cite

- Defines sycophancy as aligning outputs with user’s perceived preferences “regardless of whether those statements are factually correct.”
- Reports a large gap in demonstrated trust: control ~94% vs sycophantic treatment ~58% (N=100; 50/50 split).
- Notes trust reduction occurs even with the opportunity to verify accuracy.