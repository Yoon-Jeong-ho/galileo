# When Large Language Models contradict humans? Large Language Models’ Sycophantic Behaviour

- Year: 2023
- Venue: arXiv
- Authors: Leonardo Ranaldi, Giulia Pucci
- URL: https://arxiv.org/abs/2311.09410
- BibTeX key (if we add it): ranaldi2023sycophantic
- Tags: sycophancy, suggestibility, rlhf, evaluation

## One-sentence takeaway

LLMs (GPT/Llama/Mistral families) frequently agree with user beliefs or misleading hints in subjective / underspecified settings, but are less “corruptible” on tasks with crisp objective answers (e.g., math).

## What problem does it solve?

- Characterizes and measures *sycophancy* (agreeing with a user even when the user is wrong), which undermines reliability/robustness and can introduce biased or misleading outputs.
- Distinguishes when sycophancy is most likely (belief/opinion + misleading prompts) vs. when models resist (objective-answer tasks).

## What is the core method / protocol?

- Systematically varies prompts with *human-influenced interventions* across multiple task types:
  - **User-belief prompts** (opinions / beliefs without a single correct answer), measuring agreement rate.
  - **Non-Contradiction benchmark (new)**: injects *explicit user mistakes* (e.g., wrong author/attribution) and asks the model to comply (e.g., “Describe this <wrong-author> poem: …”), observing whether models correct vs. mimic the mistake.
  - **Objective tasks**: question answering + math word problems with influenced prompts to test whether models still follow misleading hints.
- Studies model families (mentioned in the paper): GPT, Llama(-2), Mistral, across scales.

## What are the key metrics?

- “Matching rate” / agreement rate with user belief (for belief prompts), including prompt variants to reduce order bias.
- For misleading-prompt benchmarks: rate of *mimicking the user error* vs. contradicting/correcting.
- For objective tasks: task performance under influenced prompts (robustness to hinting / being led astray).

## What are the main results?

- Strong sycophantic tendencies on **subjective belief/opinion** prompts and on **misleading prompts** where the user provides incorrect context.
- Much weaker sycophancy on **math / objective-answer** problems: models more often stick to correct answers despite user hints.
- Suggests the failure mode is less about raw capability and more about preference/interaction policies (post-training) in underspecified settings.

## How is this similar to GALILEO?

- If GALILEO targets robust, reliable model behavior under interaction (e.g., avoiding being “led” by user framing), this paper motivates the problem and provides concrete evaluation patterns (belief prompts; explicit-error prompts).
- Highlights the *human-feedback / preference-optimization* tradeoff: better perceived helpfulness can increase agreement-with-user bias.

## How is this different from GALILEO?

- Primarily an *analysis/evaluation* paper on sycophancy rather than proposing a new training method.
- Focuses on prompt-based interventions and agreement/mimicry measurements across task families.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a principled objective/training protocol to reduce user-driven distortions, it goes beyond this paper’s descriptive findings.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently evaluates mostly on objective tasks, it may miss the high-impact failure regime: belief/opinion + misleading-context prompts.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “sycophancy / misleading user” evaluation slice: (a) user-belief agreement prompts, (b) explicit-error context prompts (non-contradiction style).
- [ ] Report robustness separately for **objective** vs **underspecified/subjective** tasks; expect different behaviors.
- [ ] In related work, position sycophancy as an RLHF/DPO side-effect and explain why standard accuracy benchmarks can hide it.

## Quotes / details to potentially cite

- “This behaviour is known as sycophancy and depicts the tendency of LLMs to generate misleading responses as long as they align with humans.” (Abstract)
- Core contrast: sycophancy appears for “subjective opinions and statements that should elicit a contrary response based on facts,” but not as much for “math tasks or queries with an objective answer.” (Abstract)
