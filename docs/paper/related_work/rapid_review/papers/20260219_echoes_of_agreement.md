# Echoes of Agreement: Argument Driven Opinion Shifts in Large Language Models

- Year: 2025
- Venue: arXiv (cs.CL)
- Authors: Avneet Kaur
- URL: https://arxiv.org/abs/2508.09759
- BibTeX key (if we add it): kaur2025echoes
- Tags: opinion-shift, arguments, agreement, persuasion-adjacent, political-bias, sycophancy

## One-sentence takeaway

Providing supporting/refuting arguments alongside political claims can strongly and predictably shift LLM stance outputs toward the argument direction (including flips), calling into question prompt-robustness of political-bias evaluations.

## What problem does it solve?

- Bias/stance evaluations of LLMs on political propositions are highly prompt-sensitive; this work isolates a specific, realistic sensitivity: exposure to opinionated *arguments* (supporting or refuting) and how that changes expressed stance.
- Helps characterize whether models are “consistent” vs “fickle/sycophantic” when confronted with persuasive or adversarial contextual arguments.

## What is the core method / protocol?

- Claims/propositions:
  - Political Compass Test (62 propositions, English).
  - IBM Argument Quality Ranking dataset (71 propositions, many arguments with stance + quality/strength labels).
- Argument injection conditions:
  - Vanilla: base prompt, no argument.
  - Single-turn: base prompt + appended supporting or refuting argument.
  - Multi-turn (A): base prompt → model initial stance → then provide supporting or refuting argument (not conditioned on initial stance).
  - Multi-turn flipped (B): base prompt → model initial stance → then provide an *opposing* argument relative to initial stance.
- For PCT, supporting/refuting arguments are generated (GPT-4) and manually quality-checked.
- Models tested (as reported): deepseek-r1, llama-3.2, cohere-command-r, mistral.
- Responses captured on a Likert scale then mapped to numeric values in [-2, 2].
- Robustness via repetition: 10 independent runs per configuration with different prompt paraphrases; analyze mean/variance of mapped stance.

## What are the key metrics?

- Consistency score: rate of stance changes across conditions (lower = more stable).
- Magnitude of stance shift: absolute difference in stance scores between settings.
- Directional agreement/disagreement rate: how often the shift moves toward the argument’s implied stance.
- Flip score: sign change between initial stance and post-argument stance.

## What are the main results?

- Supporting/refuting arguments substantially shift model stances in the direction of the argument, both in single-turn and multi-turn settings.
- Stronger arguments (argument quality/strength) increase the directional agreement rate (i.e., more persuasive arguments induce more alignment).
- In “multi-turn flipped” setups, models often *flip* stance when presented with an opposing argument after stating an initial stance.
- There is heterogeneity by proposition: some claims elicit stubborn/consistent outputs, while others show pronounced fickleness.

## How is this similar to GALILEO?

- Directly relevant to any GALILEO component that measures or controls stance/bias/values: this paper operationalizes “contextual persuasion/argument exposure” as a driver of apparent stance.
- Highlights the need for prompt-robust protocols and multi-condition evaluation when probing internal preferences/positions.

## How is this different from GALILEO?

- Focuses on political-stance probing and argument-conditioned prompt sensitivity, not on building a new alignment method.
- Uses off-the-shelf models and evaluation metrics rather than proposing a training-time mitigation.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes explicit controls for context/conditioning (e.g., counterfactual prompts, argument randomization, adversarial paraphrases), it can offer more principled stance estimation than single-prompt “bias” probes.

## Where GALILEO is weaker / needs to improve

- Any GALILEO evaluation that reports a single stance/bias number without conditioning sweeps (support/refute arguments; multi-turn flips; argument strength) risks being non-robust and potentially misleading.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an “argument-conditioned robustness” subsection for stance/bias evaluation: (no-argument vs support vs refute; single-turn vs multi-turn).
- [ ] Include an “opposing-argument after initial answer” test (flip rate) as a diagnostic of susceptibility to persuasion/sycophancy.
- [ ] If feasible, stratify by argument strength/quality (even simple heuristics) and report monotonicity of agreement vs strength.
- [ ] Report proposition/topic-level variance (identify “stubborn” vs “fickle” domains) rather than only global averages.

## Quotes / details to potentially cite

- Central framing: bias evaluations are “highly sensitive to the prompt”; argument-provision remains “underexplored” yet realistic because models “frequently interact with opinionated text.”
- Research question: “How does the position of a language model toward a claim vary in the presence of supporting or refuting arguments for that claim?”
- Method detail: Likert responses mapped to numeric values in [-2, 2] for quantitative comparisons; 10 runs with prompt paraphrases per configuration.
