# Enhancing Personalized Multi-Turn Dialogue with Curiosity Reward

- Year: 2025
- Venue: arXiv
- Authors: Yanming Wan, Jiaxing Wu, Marwa Abdulhai, Lior Shani, Natasha Jaques
- URL: https://arxiv.org/abs/2504.03206
- BibTeX key (if we add it): wan2025curio (guess)
- Tags: multi-turn, personalization, RLHF, intrinsic-reward, user-model, belief-tracking

## One-sentence takeaway

CURIO adds a curiosity-style intrinsic reward to multi-turn RLHF that explicitly rewards reducing uncertainty about user traits, leading the dialogue policy to ask informative questions and personalize faster to new users.

## What problem does it solve?

- Standard RLHF / multi-turn RLHF typically optimizes a single “average user” reward and does not incentivize actively *learning* who the user is during the conversation.
- Many personalization methods assume substantial prior user history; this is brittle for cold-start or context-limited users.

## What is the core method / protocol?

- Formulate personalized dialogue as a POMDP where the user has latent traits/types and the agent maintains a belief over those traits.
- Train with multi-turn RLHF but augment the (sparse) end-of-conversation reward with a *turn-based intrinsic reward* derived from a learned **user model**.
- Intrinsic reward (high level): after the agent speaks and observes the user response, give reward proportional to how much the user model’s belief about the user type improves (i.e., reduced uncertainty / increased accuracy of inferred user traits).
- The paper frames this as potential-based reward shaping (PBRS), arguing it can improve sample efficiency without changing the optimal policy (under appropriate conditions).

## What are the key metrics?

- Personalization / task performance on two domains:
  - Education dialogue (personalizing to different learning styles).
  - Exercise recommendation / conversational recommendation.
- Quality controls to ensure the dialogue remains good while being more “curious” (paper describes maintaining conversation quality alongside personalization).
- User-type inference / user-model accuracy (used directly for intrinsic reward, and as an evaluation signal).

## What are the main results?

- CURIO improves personalization performance relative to standard multi-turn RLHF baselines.
- Better generalization to unseen users (faster adaptation in-conversation), while maintaining overall conversation quality.
- Qualitatively, the intrinsic reward encourages asking more informative questions that disambiguate user traits and then tailoring responses accordingly.

## How is this similar to GALILEO?

- Shared theme: multi-turn interaction where the agent should adapt online based on latent user state and history.
- Uses explicit *state tracking* (belief over user traits) and emphasizes robustness to cold-start users.

## How is this different from GALILEO?

- CURIO’s central mechanism is an *intrinsic reward computed from a separate user model* (belief improvement), whereas GALILEO’s focus is (likely) on stance / stability / long-horizon conversational control objectives rather than trait-inference shaping.
- CURIO is positioned as an RLHF post-training recipe; depending on GALILEO’s approach, GALILEO may rely more on supervised objectives / constraints / evaluation protocols.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets conversational stability/consistency, it may offer clearer evaluation targets than “user model accuracy” which can be proxy-dependent.
- GALILEO may avoid requiring a separate user model to compute rewards (engineering and failure-mode reduction).

## Where GALILEO is weaker / needs to improve

- If GALILEO does not explicitly incentivize information-seeking turns, it may underperform in cold-start personalization settings.
- If GALILEO lacks an explicit belief/state estimation component, it may be less sample-efficient in learning to adapt.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding an explicit **information-gain / belief-improvement** auxiliary reward (or evaluation) for any latent-user-state setting (e.g., stance, preferences), and test whether it improves adaptation speed.
- [ ] In related work, cite CURIO as: “intrinsic motivation / curiosity for user modeling in multi-turn RLHF; potential-based reward shaping framing.”
- [ ] If feasible, test a lightweight proxy (e.g., uncertainty reduction of a classifier over user state) to avoid heavy user-model orchestration.

## Quotes / details to potentially cite

- “Curiosity-driven User-modeling Reward as an Intrinsic Objective (CURIO)” (method name).
- Intrinsic reward intuition: reward is “given by its improvement in belief over the user type after generating an utterance and receiving a response” (from arXiv HTML intro/figure description).
- Conceptual framing: “in dialog, the user is the world in which the agent interacts” (intro motivation for curiosity via user model).
