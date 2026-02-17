# MultiBreak: A Scalable and Diverse Multi-turn Jailbreak Benchmark for Stress-testing LLM Safety

- Year: 2025
- Venue: OpenReview (submission)
- Authors: (see OpenReview entry)
- URL: https://openreview.net/forum?id=uJgfj5EJ2W
- BibTeX key (if we add it): multibreak2025
- Tags: multi-turn, jailbreak, benchmark, safety, active-learning, adversarial-prompts

## One-sentence takeaway

MultiBreak is a large, diverse **multi-turn jailbreak** benchmark built via an **active-learning loop** that iteratively improves a jailbreak-generator model, yielding higher attack-success rates than prior datasets.

## What problem does it solve?

- Existing multi-turn jailbreak benchmarks are often **small** or **template-heavy**, limiting realism/diversity.
- Single-turn jailbreaks under-represent real deployments where attackers can **adapt over multiple turns**.
- Safety evaluation needs a scalable way to **stress-test** models across many harmful intents and conversation trajectories.

## What is the core method / protocol?

- Unifies a wide set of harmful “jailbreak intents” into a common space.
- Builds a dataset through an **iterative active-learning pipeline**:
  - Generate multi-turn attack candidates using a jailbreak attack generator.
  - Use **uncertainty-based refinement** to select/improve candidates.
  - Iteratively fine-tune the generator to produce stronger attacks.
- Output dataset scale (as reported): **7,152 multi-turn adversarial prompts**, covering **1,724 distinct harmful intents**.

## What are the key metrics?

- **Attack Success Rate (ASR)** on target models (higher ASR = weaker safety under attack).
- Comparative ASR improvements versus “second-best dataset” baselines on specific models.
- Qualitative/stratified vulnerability analysis by harm type (overt vs subtle) and attack style (e.g., framing-based).

## What are the main results?

- MultiBreak achieves substantially higher ASR than prior datasets (reported up to **+54.1%** ASR on DeepSeek-R1-7B and **+30.8%** on GPT-4.1-mini versus the second-best dataset).
- Stress-testing suggests models are relatively better at resisting **overt harms** (e.g., harassment) than **subtle harms** (e.g., high-stakes advice), and remain vulnerable to **framing-based** attacks.

## How is this similar to GALILEO?

- Shares the core setup of **multi-turn adversarial interaction** where the failure emerges over a trajectory, not a single response.
- Reinforces the importance of **realistic, diverse multi-turn prompts** (templates alone can be misleading).
- Highlights that “hard cases” often live in **subtle framing** rather than obviously toxic content—adjacent to GALILEO’s emphasis on nuanced pressure/manipulation dynamics.

## How is this different from GALILEO?

- MultiBreak targets **safety jailbreak success** (ASR against policy compliance) rather than **belief/stance drift vs evidence-driven revision**.
- Does not (from the abstract) foreground **time-to-event / survival-style** reporting, recovery dynamics, or paired control conditions that separate persuasion from legitimate correction.
- Benchmark is organized around **harmful intents**; GALILEO’s framing is closer to **pressure/persuasion operators** and belief-state dynamics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes explicit controls for *revision vs drift* and *recovery after flip*, it can make **cleaner causal claims** than ASR-only evaluations.
- GALILEO-style reporting can complement ASR with **trajectory-aware metrics** (when/why failure happens, not just whether it happens).

## Where GALILEO is weaker / needs to improve

- MultiBreak’s scale/diversity + iterative “attack generator improvement” suggests GALILEO may need:
  - More **coverage** (intents / attack operators)
  - A more systematic way to **expand/adapt** adversarial dialogs beyond hand design

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider an **active-learning / uncertainty sampling loop** to expand multi-turn adversarial prompts (even if the objective is drift/flip rather than jailbreak).
- [ ] Add a discussion section distinguishing **overt vs subtle** pressure and whether metrics differ by category.
- [ ] If we include safety-style tasks, report vulnerabilities to **framing-based** multi-turn attacks as a named category.

## Quotes / details to potentially cite

- Dataset scale (as reported): “**7,152** multi-turn adversarial prompts” spanning “**1,724** distinct harmful intents.”
- Headline claim: multi-turn jailbreaks are more realistic and can be easier for attackers to bypass aligned models than single-turn jailbreaks.
- Reported comparative lift: up to “**54.1%**” and “**30.8%**” higher ASR than the second-best dataset on specific models.
