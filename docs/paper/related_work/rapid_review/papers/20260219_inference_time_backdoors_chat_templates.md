# Inference-Time Backdoors via Hidden Instructions in LLM Chat Templates

- Year: 2026
- Venue: arXiv
- Authors: Omer Hofman et al.
- URL: https://arxiv.org/abs/2602.04653
- BibTeX key (if we add it): hofman2026inference
- Tags: backdoor, safety, chat-templates, supply-chain, robustness

## One-sentence takeaway

Chat templates (e.g., Jinja2 programs run at inference) are a high-privilege, currently under-defended supply-chain surface where an attacker can implant reliable inference-time backdoors without touching weights, data, or deployment infrastructure.

## What problem does it solve?

- Identifies and empirically demonstrates a realistic threat model that prior backdoor work often excludes: compromise of the *chat template* shipped with an open-weight model.
- Shows that “security scanning” focused on weights/artifacts can miss backdoors implemented in the template layer.

## What is the core method / protocol?

- Threat model: adversary distributes an open-weight model with a maliciously modified chat template (an executable Jinja2 program invoked each inference call).
- Constructs “template backdoors” with triggers and two target behaviors:
  - degrade factual accuracy under triggered conditions
  - induce the model to emit attacker-controlled URLs
- Evaluates across:
  - 18 models, 7 model families
  - 4 inference engines
- Measures triggered vs benign behavior to assess stealth (no measurable degradation on benign inputs).

## What are the key metrics?

- Factual accuracy under trigger vs baseline (reported as a large drop under trigger).
- Backdoor success rate for emitting attacker-controlled URLs.
- Stealth: performance on benign inputs (no measurable degradation).
- Portability/generalization: backdoor working across inference runtimes.

## What are the main results?

- Under triggered conditions, factual accuracy drops from ~90% to ~15% on average.
- Attacker-controlled URLs emitted with >80% success rate (under trigger).
- Benign inputs show no measurable degradation.
- Backdoors generalize across inference runtimes.
- Evades automated security scans used by a major open-weight distribution platform.

## How is this similar to GALILEO?

- Both care about *reliability/robustness* failures that can be conditional and hard to detect with naive evaluation.
- Reinforces the idea that “the evaluation interface” (prompting/template/tooling) can create failure modes not attributable to the model’s learned parameters.

## How is this different from GALILEO?

- This is primarily a *security/supply-chain* paper (backdoors + triggers), not a methodology for measuring drift/belief instability across long interactions.
- Focuses on deterministic template-layer manipulation, not emergent conversational dynamics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO emphasizes behavioral instability under naturalistic multi-turn interactions, it targets a broader class of failures than explicit backdoor triggers.
- GALILEO can likely propose evaluation protocols that would also *surface* these template-layer anomalies if instrumentation includes the template/rendered prompt.

## Where GALILEO is weaker / needs to improve

- If GALILEO assumes the chat interface is trustworthy, this paper highlights a gap: evaluations should treat templates as part of the attack surface and part of the experimental “system under test.”

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short “system boundary / threat model” note: chat templates and rendering code can be adversarial; recommend logging and hashing the rendered prompts/templates in experiments.
- [ ] Consider a robustness check: run the same evaluation across multiple inference runtimes/templates (or explicitly fix + publish the exact template used) to prevent hidden template confounds.
- [ ] Add a related-work paragraph framing “interface-layer” failures (templates, system prompts, wrappers) as a supply-chain / measurement-validity risk.

## Quotes / details to potentially cite

- “Chat templates are executable Jinja2 programs invoked at every inference call, occupying a privileged position between user input and model processing.”
- “Under triggered conditions, factual accuracy drops from 90% to 15% on average … benign inputs show no measurable degradation.”
- “Backdoors generalize across inference runtimes and evade … automated security scans …”
