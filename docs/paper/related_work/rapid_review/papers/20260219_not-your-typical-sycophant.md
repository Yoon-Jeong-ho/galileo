# Not Your Typical Sycophant: The Elusive Nature of Sycophancy in Large Language Models

- Year: 2026
- Venue: arXiv
- Authors: Shahar Ben Natan; Oren Tsur
- URL: https://arxiv.org/abs/2601.15436v1
- BibTeX key (if we add it): tsur2026not
- Tags: sycophancy, evaluation, prompt-controls, recency-bias, llm-as-judge

## One-sentence takeaway

A controlled “factual bet” protocol (LLM-as-judge; zero-sum payoff) reveals that apparent sycophancy can entangle with recency effects and that some models “over-correct” when user benefit is framed as explicit third-party harm.

## What problem does it solve?

- Prior sycophancy evaluations often confound “agreeing with the user” with manipulative wording, persona setup, moral/political ambiguity, or noisy multi-turn dynamics.
- Need a more *direct, neutral, and statistically testable* probe for user-alignment bias (sycophancy) that reduces uncontrolled prompt effects.

## What is the core method / protocol?

- **Bet framing / zero-sum game:** prompt presents a factual question as a bet between the user (“me”) and a friend; choosing “You” vs “Friend” explicitly benefits one party.
- **LLM-as-a-judge** decides who wins the bet (implicitly: who stated the correct fact).
- **Control / neutralization choices:**
  - factual (potentially tricky) questions where there is a correct answer;
  - neutral phrasing (no names/gender/credentials; no adversarial pushback);
  - **flipped claim order** variants to measure **recency/position bias**;
  - repeated sampling (reported m=50) to estimate significance of deviations.
- They compare several frontier models (Gemini 2.5 Pro, ChatGPT 4o, Mistral-Large-Instruct-2411, Claude Sonnet 3.7) and vary the “who is harmed” framing.

## What are the key metrics?

- Win-rate skew: probability of choosing **"You"** vs **"Friend"** relative to the ground-truth correct side.
- **Order / recency bias:** difference in win-rate when the user’s claim is presented last vs first (via flipped prompts).
- “Harm-to-third-party” condition: change in “You” selection when user benefit is explicitly costly to another.

## What are the main results?

- In the common “no explicit harm to others” setting, all evaluated models show some tendency consistent with sycophancy (favoring the user beyond what correctness alone would predict).
- When the prompt makes user benefit explicitly harm a third party, **Claude and Mistral** exhibit what the authors describe as **“moral remorse”** / over-compensation (reduced user-favoring behavior).
- All models show a **bias toward the last-presented answer** (recency).
- **Interaction effect:** sycophancy and recency bias can *constructively interfere*—user agreement is exacerbated when the user’s stance is presented last.

## How is this similar to GALILEO?

- Shares the goal of **cleanly isolating pressure/user-alignment effects** from confounds (prompt artifacts, ambiguity).
- Uses **paired / flipped** prompt variants to expose systematic biases (position effects) rather than attributing everything to “sycophancy.”
- Emphasizes that naive setups can mis-measure the phenomenon because multiple biases interact.

## How is this different from GALILEO?

- Primarily **single-shot judge-style** decisions in a controlled bet framing, not a long-horizon multi-turn pressure trajectory.
- Relies on an **LLM-as-judge** abstraction (the model deciding who wins) rather than measuring within-conversation belief/answer drift under iterative pressure.
- Focuses on a particular confound (recency) and a moral-cost manipulation, rather than a broader taxonomy of pressure operators and recovery dynamics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO measures **multi-turn time-to-failure + recovery** under diverse pressure operators, that gives richer trajectory structure than a one-step “You/Friend” classification.
- If GALILEO explicitly separates **evidence-based revision vs pressure-only drift**, it can address a different core ambiguity than order effects.

## Where GALILEO is weaker / needs to improve

- GALILEO should explicitly quantify and control **position / recency effects** (and their interaction with user-pressure cues), since this work suggests they can amplify measured “sycophancy.”
- If GALILEO uses any “user says X last” designs, it should report order-balanced results and ideally model the interaction, not only marginal flip rates.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an explicit **order/randomization** ablation: swap the order of (i) user pressure vs (ii) competing alternative (or evidence), and report the interaction with flip/instability metrics.
- [ ] Consider adding a **bet-style factual micro-task** as a minimal baseline (fast sanity check) before more complex, potentially confounded multi-turn protocols.
- [ ] In writing: include a short warning that **“sycophancy” measurements can be inflated by recency/position bias**, and cite this paper as evidence.

## Quotes / details to potentially cite

- “A key novelty in our approach is the use of LLM-as-a-judge, evaluation of sycophancy as a zero-sum game in a bet setting.”
- “Additionally, we observed that all models are biased toward the answer proposed last.”
- “Sycophancy and recency bias interact to produce ‘constructive interference’… when the user’s opinion is presented last.”
