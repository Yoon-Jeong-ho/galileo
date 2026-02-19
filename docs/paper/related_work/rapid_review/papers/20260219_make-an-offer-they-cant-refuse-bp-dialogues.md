# Make an Offer They Can't Refuse: Grounding Bayesian Persuasion in Real-World Dialogues without Pre-Commitment

- Year: 2025
- Venue: arXiv (under review; formatted as AAMAS 2026 in HTML)
- Authors: Buwei He; Yang Liu; Zhaowei Zhang; Zixia Jia; Huijia Wu; Zhaofeng He; Zilong Zheng; Yipeng Kang
- URL: https://arxiv.org/abs/2510.13387v2
- BibTeX key (if we add it): he2025offer
- Tags: bayesian-persuasion, information-design, llm, dialogue, commitment

## One-sentence takeaway

They show a practical way to "do Bayesian persuasion in one turn" with LLMs by having the persuader explicitly narrate an honesty/dishonesty type model (a commitment surrogate), enabling the receiver to perform a Bayesian update and increasing persuasion success vs non-BP baselines.

## What problem does it solve?

- Classic Bayesian Persuasion (BP) assumes a pre-committed signaling scheme that is common knowledge; in real dialogue, that commitment is not available.
- Prior LLM persuasion work often (a) does not explicitly exploit information asymmetry, or (b) bakes the BP schema into the receiver prompt (a strong pre-commitment / prompt-engineering assumption).
- Goal: implement BP-like information design *within* natural language, in a single-turn interaction, without relying on external pre-commitment.

## What is the core method / protocol?

- Setup: one-turn Sender (S) and Receiver (R). S observes world state omega; R has a prior and chooses action a in {Accept, Reject}.
- Key mechanism: "type-induced commitment-communication".
  - The sender's message is structured into four components:
    - m_basic: background on states / framing
    - m_type: a narrative about the sender's *type* (e.g., Honest vs Dishonest) that induces a belief p(theta) in the receiver
    - m_des: a description of the observed state
    - m_inf: explicit inference / utility argument guiding the receiver's decision
  - The receiver uses the induced p(theta) to form an *effective* signaling schema as a mixture of base policies (truthful vs persuasion-maximizing), and then applies Bayes rule to update belief about omega.
- Two natural-language realizations:
  - SFNL (semi-formal): mixes explicit probabilities / expected utility calculations with narrative.
  - FNL (fully natural): expresses the same reasoning in fluent language (uncertainty, confidence, cost-benefit) without explicit math.
- Two "views" for deployment realism:
  - Explicit view: provide the scenario's Bayesian setup (priors, utilities) externally to the persuader LLM.
  - Self-derived view: the persuader infers the Bayesian setup from the situation (more realistic / ambiguous).

## What are the key metrics?

- Primary: persuasion success / acceptance rate of the receiver (P(a = Accept | m)).
- Reported qualitative axes (from abstract/intro):
  - Credibility / logical coherence (SFNL tends to be stronger)
  - Emotional resonance / robustness in naturalistic conversation (FNL tends to be stronger)
- Evaluation includes multiple receiver types:
  - LLM receivers (different prompts / fine-tuning)
  - human participants

## What are the main results?

- BP-guided prompting increases persuasion success rates over non-BP baselines across a diverse receiver set.
- Tradeoff between variants:
  - SFNL: more credible and logically coherent.
  - FNL: more emotionally resonant and more robust in naturalistic settings.
- With supervised fine-tuning, smaller models can reach BP performance comparable to larger models (suggesting the "BP-in-NL" behavior is teachable).

## How is this similar to GALILEO?

- If GALILEO is about strategic communication / decision shaping / interaction protocols, this is a concrete instance of:
  - making latent assumptions explicit in language (commitment / schema communication)
  - structuring messages into interpretable components (background, assumptions, observation, inference)
  - optimizing downstream decision quality / acceptance under information asymmetry.

## How is this different from GALILEO?

- Focus is explicitly on Bayesian persuasion as the normative model; commitment is implemented via a "sender types" story, not via verifiable protocol or repeated-game credibility.
- One-turn, rhetorical framing focus; not primarily about multi-turn dialogue, tool use, or verifiable evidence exchange.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO emphasizes verifiability, calibration, or evidence-grounding, it can avoid the "dishonest type" rhetorical framing that may incentivize manipulative persuasion.
- If GALILEO operates multi-turn, it can use iterative clarification / evidence presentation rather than relying on a single crafted message.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks an explicit information-asymmetry / commitment model, this paper suggests a simple language-level mechanism to make the receiver's inference process more controlled and predictable.
- If GALILEO evaluation does not include diverse receivers (different LLM prompts + humans), this work's evaluation breadth is a useful benchmark.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an ablation inspired by (m_basic, m_type, m_des, m_inf): measure which components move acceptance/decision the most.
- [ ] Compare a "semi-formal" vs "fully natural" variant for GALILEO outputs; evaluate credibility vs robustness tradeoff.
- [ ] If relevant, design a safer analogue of m_type: communicate *uncertainty/calibration* or *evidence reliability* instead of honest/dishonest types.
- [ ] Extend evaluation to heterogeneous receivers (different system prompts / instruction levels) to test robustness.

## Quotes / details to potentially cite

- "commitment-communication mechanism" via explicitly narrating sender types (honest/dishonest) to induce a belief over types and thereby an effective information schema.
- Composite signal structure: m_basic + m_type + m_des + m_inf.
- Claimed findings (abstract): BP > non-BP; SFNL more coherent/credible; FNL more emotionally resonant/robust; SFT lets smaller models match larger ones.
