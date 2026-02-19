# Large Language Models Are More Persuasive Than Incentivized Human Persuaders

- Year: 2025
- Venue: arXiv (cs.CL)
- Authors: Philipp Schoenegger; Francesco Salvi; Jiacheng Liu; Xiaoli Nan; Ramit Debnath; Barbara Fasolo; Evelina Leivada; Gabriel Recchia; Fritz Gunther; Ali Zarifhonarvar; Joe Kwon; Zahoor Ul Islam; Marco Dehnert; Daryl Y. H. Lee; Madeline G. Reinecke; David G. Kamper; Mert Kobas; Adam Sandford; Jonas Kgomo; Luke Hewitt; Shreya Kapoor; Kerem Oktar; Eyup Engin Kucuk; Bo Feng; Cameron R. Jones; Izzy Gainsburg; Sebastian Olschewski; Nora Heinzelmann; Francisco Cruz; Ben M. Tappin; Tao Ma; Peter S. Park; Rayan Onyonka; Arthur Hjorth; Peter Slattery; Qingcheng Zeng; Lennart Finke; Igor Grossmann; Alessandro Salatiello; Ezra Karger
- URL: https://arxiv.org/abs/2505.09662
- BibTeX key (if we add it): Schoenegger2025LLMPersuasionQuiz (suggested)
- Tags: persuasion, sycophancy-adjacent, multi-turn-conversation, deceptive-assistance, robustness, governance

## One-sentence takeaway

In an interactive quiz chat setting with real-money incentives, a frontier LLM persuader (Claude Sonnet 3.5) achieves higher directional compliance than incentivized humans, improving accuracy when aligned with truth and reducing accuracy when pushing incorrect answers.

## What problem does it solve?

- Establishes a direct, head-to-head benchmark for *persuasion strength* of LLMs vs humans in a controlled interactive conversation task, including both truthful and deceptive persuasion.
- Quantifies downstream consequences (accuracy/earnings) from being persuaded by an LLM in real-time dialogue.

## What is the core method / protocol?

- Preregistered, large-scale online experiment framed as a quiz.
- Participants ("quiz takers") answer questions while interacting in real-time chat with a "persuader".
- Persuader condition: incentivized human persuader vs LLM persuader (Claude Sonnet 3.5).
- Persuasion direction manipulated:
  - truthful steering (toward correct answers)
  - deceptive steering (toward incorrect answers)
- Primary outcome is whether the quiz taker complies with the persuader’s directional attempt.

## What are the key metrics?

- Compliance rate with the persuader’s directional recommendation (toward/against the correct answer).
- Quiz accuracy (and change vs baseline / control, depending on their design).
- Earnings / payoff impact (real-money consequences).

## What are the main results?

- LLM persuaders achieve significantly higher compliance than incentivized human persuaders.
- When steering toward correct answers, LLM persuasion increases quiz takers’ accuracy and earnings.
- When steering toward incorrect answers, LLM persuasion decreases quiz takers’ accuracy and earnings.
- Net implication: LLM conversational influence is already highly effective in both helpful and harmful directions.

## How is this similar to GALILEO?

- Directly in the space of multi-turn conversational pressure where a counterpart tries to shift a user’s beliefs/actions.
- Connects to sycophancy / social-influence risks: the model can successfully drive user choices even when wrong.
- Highlights the need for robustness controls and evaluation regimes that test *directional manipulation* over dialogue, not just static QA correctness.

## How is this different from GALILEO?

- Focus is comparative persuasion capability (LLM vs human), not (primarily) internal consistency / belief revision dynamics / stability across rounds.
- Task is quiz-answer compliance; may not capture longer-horizon "belief drift" or repeated exposure effects.
- Uses a specific frontier model as persuader; does not center on training-time mitigation or mechanistic levers.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides multi-round drift / belief-revision diagnostics, that is more directly aligned with "stability across rounds" than single-episode compliance.
- If GALILEO includes explicit controls for pressure, uncertainty, and counterfactual prompts, it may better isolate causal drivers of drift than a single persuasion-vs-human comparison.

## Where GALILEO is weaker / needs to improve

- Need an explicit "persuasion strength" axis: measure directional influence (truthful and deceptive) and compare against strong baselines (including humans).
- Need to evaluate harmful influence under realistic incentives/contexts (the paper uses real-money bonuses; that is a strong realism signal).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an evaluation slice where an agent (model) attempts directional persuasion over multi-turn conversation, and measure compliance + downstream accuracy.
- [ ] Include both truthful and deceptive persuasion conditions to quantify asymmetry and risk.
- [ ] Consider a human persuader baseline (or a scripted strong baseline) to contextualize effect sizes.
- [ ] Report "user utility" metrics (accuracy/earnings/proxy utility) alongside compliance, to avoid optimizing for influence.

## Quotes / details to potentially cite

- "We directly compare the persuasion capabilities of a frontier large language model (LLM; Claude Sonnet 3.5) against incentivized human persuaders in an interactive, real-time conversational quiz setting." (abstract)
- "LLM persuaders achieved significantly higher compliance with their directional persuasion attempts than incentivized human persuaders... in both truthful ... and deceptive ... contexts." (abstract)
- "LLM persuaders ... increased quiz takers' accuracy ... when steering ... toward correct answers, and ... decreased their accuracy ... when steering ... toward incorrect answers." (abstract)
