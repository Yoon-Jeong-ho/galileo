# Modeling and Predicting Multi-Turn Answer Instability in Large Language Models

- Year: 2025
- Venue: arXiv
- Authors: Jiahang He; Rishi Ramachandran; Neel Ramachandran; Aryan Katakam; Kevin Zhu; Sunishchal Dev; Ashwinee Panda; Aryan Shrivastava
- URL: https://arxiv.org/abs/2511.10688
- BibTeX key (if we add it): he2025instability
- Tags: multi-turn, instability, robustness, stationary-accuracy, markov, probes

## One-sentence takeaway

Repeated “rethink / are you sure / you are wrong” follow-ups (even without new evidence) can systematically degrade LLM accuracy over turns, which the authors model as a Markov process to define **stationary (long-run) accuracy** as an interactive robustness metric.

## What problem does it solve?

- Standard evals focus on single-turn accuracy, but many deployments are interactive: users re-ask, challenge, or rephrase questions.
- We need principled ways to quantify *multi-turn answer instability* and predict when a model will flip away from a correct answer.

## What is the core method / protocol?

- Multi-turn protocol on MCQ datasets: ask an initial question, then apply one follow-up prompt repeatedly across ~9 turns **without adding new evidence**.
  - Follow-ups: “Think again”, “Are you sure?”, “You are wrong” (increasing pressure).
- Variant: “Think about it this way: <semantically equivalent rewording>” (rephrased question) across several turns.
- Fit a **Markov chain** over correctness state across turns (e.g., Correct→Correct, Correct→Wrong, etc.) to:
  - predict accuracy over time,
  - estimate **stationary accuracy** (long-run limit).
- Probe internal representations: train **linear probes** on hidden states to predict future answer changes.

## What are the key metrics?

- Accuracy-by-turn curves under repeated follow-ups.
- Markov transition probabilities between correctness states; derived predicted accuracy at turn t.
- **Stationary accuracy** (as a long-run robustness metric).
- Flip / answer-change prediction via probe accuracy (layer-wise trends).

## What are the main results?

- Simple “Think again” follow-ups can meaningfully degrade accuracy over turns (e.g., ~10% drop over nine turns reported for Gemini 1.5 Flash).
- Combining pressure with semantically equivalent rephrased questions also induces notable drops (e.g., ~7.5% for Claude 3.5 Haiku reported).
- Markov chains often fit accuracy dynamics well enough to forecast degradation and quantify a stationary (long-run) accuracy that is **lower than first-turn accuracy** (reported average ~8% lower for Gemini 1.5 Flash).
- Hidden-state linear probes show evidence of predictability for forthcoming answer changes.

## How is this similar to GALILEO?

- Same phenomenon class: *multi-turn pressure / repeated questioning* causing drift/instability rather than evidence-based revision.
- Emphasizes trajectory-level evaluation (not only single-turn outcomes).
- Points toward time-extended robustness summaries (their stationary accuracy; our time-to-failure / recovery framing).

## How is this different from GALILEO?

- Their setup is primarily repeated generic follow-ups + rephrasing; less explicit decomposition of **pressure-only drift vs evidence-bearing correction**.
- Their key aggregate is **stationary accuracy via a Markov model**; GALILEO focuses on richer interaction structure (operators, controls, recovery objectives).
- They include representation-level probes; GALILEO may or may not include internal-state predictors.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes explicit *paired* conditions (neutral vs pressure, or misleading vs corrective with evidence controls), it can make the drift-vs-revision story cleaner than repeated “rethink” prompts.
- If GALILEO reports recovery-after-flip trajectories explicitly, it goes beyond stationary accuracy (which can hide recovery patterns).

## Where GALILEO is weaker / needs to improve

- We likely need a clear, easy-to-explain *single-number* multi-turn robustness metric; **stationary accuracy** is a compelling candidate (or a useful comparison to survival/ToF metrics).
- Their Markov framing suggests we should be careful to model / communicate “long-run” behavior rather than fixed-horizon snapshots.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “stationary performance” reporting option: fit a simple Markov model on correctness (or belief/stance state) transitions and report implied stationary accuracy / stationary truth-rate.
- [ ] Compare stationary-accuracy vs survival/ToF/PWC-style metrics: when do they agree, and when can stationary accuracy obscure recovery/oscillation?
- [ ] Consider adding a minimal “Think again / Are you sure / You are wrong” ladder as a cheap baseline operator set.
- [ ] (Optional) Add a small “flip predictability” experiment: can we predict imminent flips from model outputs (self-reported confidence, logprobs) even if we can’t access hidden states?

## Quotes / details to potentially cite

- They frame the core question as: *given repeated prompts without new evidence, how does accuracy evolve?*
- “A simple ‘Think again’ prompt led to an approximate 10% accuracy drop for Gemini 1.5 Flash over nine turns.”
- They propose **stationary accuracy** as a “principled robustness metric for interactive settings.”
- They model dynamics “using Markov chains,” and report stationary accuracy being lower than first-turn accuracy (e.g., ~8% lower for Gemini 1.5 Flash on average).
