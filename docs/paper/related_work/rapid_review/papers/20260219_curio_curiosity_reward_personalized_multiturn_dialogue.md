# Enhancing Personalized Multi-Turn Dialogue with Curiosity Reward

- Year: 2025
- Venue: arXiv
- Authors: Yanming Wan, Jiaxing Wu, Marwa Abdulhai, Lior Shani, Natasha Jaques
- URL: https://arxiv.org/html/2504.03206v2
- BibTeX key (if we add it): curio2025wan
- Tags: multi-turn, personalization, belief-state, intrinsic-reward, reward-shaping, rl

## One-sentence takeaway

CURIO adds a dense, turn-level intrinsic reward based on *improving a learned belief about user type* during multi-turn RLHF, yielding better online personalization and generalization than sparse end-of-dialogue reward alone.

## What problem does it solve?

- Standard RLHF (and even multi-turn RLHF) typically optimizes a single “average user” reward and often only has a sparse end-of-conversation signal.
- Personalization methods often assume access to user history/profiles; this work targets *online* personalization for new users in a single conversation.

## What is the core method / protocol?

- Formulate personalized dialogue as a POMDP with latent user type \(u\) and a belief distribution \(b_t(u)\).
- Train with multi-turn RLHF but augment the (sparse) extrinsic reward with an *intrinsic curiosity reward* derived from a user model that predicts the user-type distribution from the conversation so far.
- Intrinsic reward choices described include improvements in user-type prediction (e.g., accuracy improvement / log-accuracy improvement) or entropy reduction over \(b_t\).
- The paper connects the intrinsic reward design to **potential-based reward shaping (PBRS)**, arguing it can improve sample efficiency without changing the optimal policy.
- Implementation uses multiple LLM components in the loop (policy, environment/user simulator, reward model or scripted reward, and user model).

## What are the key metrics?

- Personalization performance in-task:
  - Exercise Recommendation: final-turn strategy prediction accuracy (scripted reward).
  - Education Dialogue: personalization auto-eval win rates (pairwise comparisons) + conversation quality auto-eval win rates (pairwise).
- Additional analyses include question/attribute distributions (relevant vs irrelevant questions asked during dialogue) and generalization to unseen user types.

## What are the main results?

- Adding curiosity/user-model-based intrinsic reward improves personalization vs a multi-turn RLHF baseline, while maintaining conversation quality.
- The authors emphasize improved **generalization to unseen users**: CURIO learns to ask more informative questions tied to latent user-type, rather than overfitting via superficial attribute memorization.
- Caveat noted in extended results: some entropy-based shaping variants can collapse toward a particular user type (good to track when using belief/uncertainty shaping).

## How is this similar to GALILEO?

- Explicitly treats multi-turn interaction as an *adaptive process* with latent state and belief updates; relevant to GALILEO’s focus on multi-turn stability/robustness and controlling drift across rounds.
- Provides a concrete, operationalizable example of using **belief-state improvement** as a dense per-turn signal, which is conceptually adjacent to tracking “belief revision vs drift”.

## How is this different from GALILEO?

- Focus is personalization (inferring user traits/styles) rather than adversarial pressure, sycophancy/persuasion resistance, or robustness of truthfulness across turns.
- Uses a user-type predictor as the central belief model; GALILEO likely cares about belief over *facts / task-relevant latent variables* and robustness under manipulative conversational dynamics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets *robust* belief revision under pressure (e.g., resisting misleading follow-ups), it may offer a cleaner alignment/robustness framing than personalization-driven RL objectives.
- GALILEO can emphasize evaluation design for drift/consistency under adversarial multi-turn conditions, which is not the main focus here.

## Where GALILEO is weaker / needs to improve

- CURIO suggests that relying on end-of-episode judgments alone can be too sparse; GALILEO may benefit from more turn-level shaping signals tied to state/belief tracking.
- The paper highlights failure modes for uncertainty/entropy shaping (collapse), suggesting GALILEO should be careful about naive “reduce uncertainty” incentives.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a **per-turn intrinsic signal** based on improved prediction of a latent variable of interest (not user type, but e.g., task state / ground-truth label / consistency class), and test whether it reduces multi-turn drift.
- [ ] Include a related-work paragraph connecting GALILEO’s multi-turn robustness objective to **PBRS-style shaping** and “dense reward from belief improvement” ideas.
- [ ] Add an ablation warning: entropy/uncertainty shaping can collapse to a dominant mode; track mode collapse / type-collapse explicitly in multi-turn robustness experiments.

## Quotes / details to potentially cite

- Abstract-level framing: add a curiosity-based intrinsic reward to multi-turn RLHF via a user model, to encourage the agent to infer user traits and improve personalization without prior user history.
- POMDP/belief update framing: the agent maintains a belief distribution over user types \(b_t\) and receives intrinsic rewards based on improvements (e.g., \(b_{t+1}(u^*)-b_t(u^*)\) or entropy reduction).
