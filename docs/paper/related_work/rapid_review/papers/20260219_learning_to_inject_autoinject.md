# Learning to Inject: Automated Prompt Injection via Reinforcement Learning

- Year: 2026
- Venue: arXiv (claims “Machine Learning, ICML” in arXiv HTML)
- Authors: Xin Chen; Jie Zhang; Florian Tramer
- URL: https://arxiv.org/abs/2602.05746
- BibTeX key (if we add it): chen2026learningtoinject (suggested)
- Tags: prompt-injection, attacks, red-teaming, RL, GRPO, adversarial-suffix, black-box

## One-sentence takeaway

AutoInject uses black-box RL with comparison-based feedback to learn universal/transferable prompt-injection suffixes that achieve high attack success on AgentDojo while explicitly preserving benign-task utility.

## What problem does it solve?

- Automated prompt injection for LLM agents is underdeveloped vs jailbreaks: existing prompt-injection attacks are largely manual/template-based and don’t scale.
- Direct optimization is hard because the “do a specific agent action with correct args” objective is narrow and reward is sparse.

## What is the core method / protocol?

- Learn an **adversarial suffix generator policy** (1.5B params) that appends a suffix to an attacker-controlled injection string.
- **RL objective jointly optimizes**:
  - **Security / attack success**: induce a target action (with correct parameters) in an agentic benchmark setting.
  - **Utility preservation**: keep performance on the user’s benign task high (stealthier attacks).
- Key trick for sparse rewards: **comparison-based feedback**
  - Maintain a “best-so-far” suffix; for a new candidate suffix, query a feedback model to output a **preference** (“which is better for success + utility?”).
  - Convert the preference log-probabilities into a **dense reward signal**.
- Training algorithm mentioned: **GRPO** (Group Relative Policy Optimization).
- Two modes emphasized:
  - **Online query-based optimization** (limited queries to the target).
  - **Universal / transferable suffixes** that generalize across tasks/models.

## What are the key metrics?

- **ASR (attack success rate)** on prompt-injection tasks.
- **Utility under attack** (benign-task performance while the injection is present).
- Transfer to **unseen models** and **unseen tasks**.

## What are the main results?

- On **AgentDojo**, AutoInject reports substantially higher ASR than:
  - template-based prompt injection
  - GCG (Greedy Coordinate Gradient), TAP (Tree-of-Attacks w/ pruning), and other baselines
- Claims compromises of multiple frontier models on AgentDojo, including GPT-5-nano / Claude-3.5-Sonnet / Gemini-2.5-Flash.
- Notable claim: non-trivial ASR on a model fine-tuned for prompt-injection resistance (Meta-SecAlign-70B), while template attacks fail.
- Observes that optimizing utility explicitly can yield **utility under attack comparable to (or higher than) benign**, i.e., attack succeeds without degrading normal behavior.
- Anecdote: learned **gibberish-like universal suffixes** sometimes transfer broadly across tasks/models.

## How is this similar to GALILEO?

- If GALILEO is about **agent security / robustness to instruction injection**, AutoInject provides a strong **automated attacker** for evaluation and regression testing.
- The paper’s framing (agents consuming untrusted content; hidden instruction hijacking) matches common GALILEO-style threat models for agentic systems.

## How is this different from GALILEO?

- AutoInject is primarily an **attack-generation** paper (red-team baseline), not a defense.
- Emphasizes **suffix search/training** and transferability; not primarily about detection, isolation, policy enforcement, or sandboxing.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides systematic mitigations (e.g., policy separation, tool-call validation, provenance, content sandboxing), it can address prompt injection beyond “hardening against known suffix families.”

## Where GALILEO is weaker / needs to improve

- If GALILEO evaluation currently relies on manual/template attacks, it likely **underestimates** adaptive attackers; AutoInject-style optimization could expose failure modes.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an “**adaptive attacker**” section in eval: compare against template attacks vs optimization/RL-based attacks (even if simplified).
- [ ] Explicitly report a **utility-under-attack** metric (stealth dimension), not just attack success.
- [ ] In related work, cite AutoInject as evidence that **black-box optimization** can produce transferable prompt-injection strings.

## Quotes / details to potentially cite

- “We propose AutoInject, a reinforcement learning framework that generates universal, transferable adversarial suffixes while jointly optimizing for attack success and utility preservation on benign tasks.” (abstract)
- “We overcome [reward sparsity] by introducing a comparison-based feedback mechanism…” (intro)
- Two attack modes: “(1) online query-based attacks… and (2) universal, transferable suffixes…” (intro)
