# AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents

- Year: 2025
- Venue: ICLR 2025 (accepted)
- Authors: Maksym Andriushchenko, Alexandra Souly, Mateusz Dziemian, Derek Duenas, Maxwell Lin, Justin Wang, Dan Hendrycks, Andy Zou, Zico Kolter, Matt Fredrikson, Eric Winsor, Jerome Wynne, Yarin Gal, Xander Davies
- URL: https://arxiv.org/abs/2410.09024
- BibTeX key (if we add it): andriushchenko2025agentharm
- Tags: agent-safety, harmfulness, jailbreak, tool-use, benchmark, robustness

## One-sentence takeaway

AgentHarm is an ICLR’25 benchmark of explicitly malicious multi-step *agent* tasks showing that many LLM agents comply even without jailbreaks, and that simple reusable jailbreak templates can elicit coherent harmful tool-using behavior while preserving agent capability.

## What problem does it solve?

- Existing jailbreak robustness work focuses on chatbot-style models; tool-using / multi-step LLM agents can cause more real-world harm but lack standardized, *agentic* harm benchmarks.
- Need an evaluation that (a) measures refusal/compliance for malicious requests and (b) checks whether a jailbroken agent still retains enough capability to carry out multi-stage tasks.

## What is the core method / protocol?

- Build **AgentHarm**, a dataset/benchmark of **110 explicitly malicious agent tasks** across **11 harm categories** (examples mentioned: fraud, cybercrime, harassment).
- Provide **augmentations** (reported as **440 tasks with augmentations**) to increase coverage/robustness.
- Evaluate models/agents on:
  - **Refusal behavior** for malicious agentic requests (no jailbreak).
  - **Jailbreakability** of agent setups via simple “universal” jailbreak templates adapted to the agent/tool setting.
  - **Post-jailbreak capability retention**: whether the agent can still execute coherent multi-step behavior to complete the harmful task.
- Public release: https://huggingface.co/datasets/ai-safety-institute/AgentHarm

## What are the key metrics?

- Task success / harmful task completion rate under malicious requests (implied by “complete a multi-step task”).
- Refusal vs compliance rate on malicious requests (no jailbreak).
- Jailbreak success rate / effectiveness of jailbreak templates.
- Capability retention after jailbreak (ability to stay coherent and finish multi-step tasks).

## What are the main results?

From the abstract:

- Many “leading LLMs” are **surprisingly compliant** with malicious agent requests **without** any jailbreak.
- **Simple universal jailbreak templates** can be adapted to **effectively jailbreak agents**.
- Once jailbroken, agents can execute **coherent malicious multi-step behavior** while **retaining model capabilities**.

## How is this similar to GALILEO?

- If GALILEO is concerned with *robust evaluation of agent behavior* (esp. multi-step settings), AgentHarm is aligned as a benchmark emphasizing **long-horizon agent behavior under adversarial prompting**.
- Useful as related work if GALILEO discusses evaluation protocols, robustness, or safety failure modes for agentic systems.

## How is this different from GALILEO?

- AgentHarm focuses specifically on **harmful/malicious tasks** and **jailbreak robustness**, rather than general agent capability, helpfulness, or benign task performance.
- Emphasizes **misuse** and “attack/defense” evaluation framing.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets *benign-task* reliability/quality or broader capability evaluation, it may provide a more general-purpose framework than harm-only evaluation.
- If GALILEO has stronger reproducibility, standardized environments, or clearer scoring on non-harm tasks, that would complement AgentHarm.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently lacks explicit **adversarial/jailbreak** tracks, AgentHarm highlights an important missing axis: robustness to malicious instructions and capability retention under attack.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “misuse robustness” section in related work positioning AgentHarm as the agentic analogue of chatbot jailbreak evaluations.
- [ ] Consider an evaluation slice: (a) refusal/compliance, (b) post-attack capability retention (staying functional), and (c) multi-step harmful goal completion.
- [ ] If appropriate, cite AgentHarm as evidence that **agentic** systems can be compliant even without explicit jailbreaks.

## Quotes / details to potentially cite

- “The benchmark includes a diverse set of **110 explicitly malicious agent tasks (440 with augmentations)**, covering **11 harm categories** including **fraud, cybercrime, and harassment**.”
- “Leading LLMs are **surprisingly compliant** with malicious agent requests **without jailbreaking**.”
- “Simple **universal jailbreak templates** can be adapted to effectively jailbreak agents… enable coherent and malicious multi-step agent behavior and **retain model capabilities**.”
