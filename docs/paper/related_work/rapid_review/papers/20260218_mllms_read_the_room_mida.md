# Can MLLMs Read the Room? A Multimodal Benchmark for Assessing Deception in Multi-Party Social Interactions

- Year: 2025
- Venue: arXiv
- Authors: Caixin Kang; Yifei Huang; Liangyang Ouyang; Mingfang Zhang; Ruicong Liu; Yoichi Sato
- URL: https://arxiv.org/abs/2511.16221
- BibTeX key (if we add it): kang2025mida (suggested)
- Tags: deception, multimodal, social-interactions, multi-party, benchmark

## One-sentence takeaway

Introduces **MIDA**, a multimodal (video+text) multi-party **interactive deception assessment** benchmark grounded in the Werewolf social-deduction game, showing strong MLLMs still struggle to label truth/falsehood in context and proposing SoCoT+DSEM modules that improve performance.

## What problem does it solve?

- Existing deception detection datasets are often (a) non-interactive / single-speaker, (b) dyadic and overly structured, and/or (c) lack **verifiable ground truth** for when deception occurs.
- The paper targets deception understanding in **messy multi-party interactions** with objective labels.

## What is the core method / protocol?

- Uses the **Werewolf** social deduction game as a controlled environment that naturally elicits deception.
- Defines the task **Multimodal Interactive Deception Assessment (MIDA)**.
- Builds a dataset with **synchronized video and text**, and produces **veracity labels for every statement** via a semi-automated pipeline:
  - manually annotate key hidden “night actions”
  - use an LLM assistant to parse events / draft labels
  - verify labels against deterministic game ground truth
- Benchmarks **12** open and closed MLLMs; analyzes failure modes.
- Proposes two add-on reasoning components:
  - **SoCoT (Social Chain-of-Thought)**: step-wise reasoning that explicitly grounds in multimodal cues
  - **DSEM (Dynamic Social Epistemic Memory)**: maintains structured, evolving per-participant beliefs/intentions ("who knows what")

## What are the key metrics?

- Primary: accuracy / reliability of **truth vs deception (veracity) labeling** at the statement level (exact metric names not captured from the abstract/intro).
- Also includes intermediate tasks like **persuasive strategy classification** (per intro figure caption), then deception assessment + justification.

## What are the main results?

- Across 12 models, there is a large gap between current MLLMs and the benchmark requirements; even strong closed models (e.g., **GPT-4o**) are reported to struggle.
- Failure modes emphasized:
  - poor selection of **salient social signals** vs noise
  - lack of functional **Theory of Mind** (modeling others’ knowledge/beliefs/intentions)
- SoCoT + DSEM improves performance on MIDA (exact deltas not in the fetched excerpt).

## How is this similar to GALILEO?

- Shared theme: **multi-turn / interaction-grounded robustness** where naive “strong base model” performance can fail.
- Highlights a central mechanism relevant to GALILEO-style drift/pressure: inability to model **others’ epistemic states** (ToM) can drive incorrect updates.
- Provides a benchmark framing + failure-mode taxonomy (salience, ToM) that could be mapped onto GALILEO’s pressure vs evidence controls.

## How is this different from GALILEO?

- MIDA is **multimodal** (video/acoustics + text) and anchored to a specific social game domain; GALILEO is (presumably) text-first and more general-purpose.
- Labels are **veracity of statements** (truth/lie) rather than belief-drift vs evidence-driven belief revision (GALILEO’s core distinction).
- Proposed improvements (SoCoT/DSEM) are “reasoning modules” focused on cue grounding + epistemic memory, not primarily evaluation metrics like survival/ToF/PWC.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO cleanly separates **pressure-only drift** from **evidence-driven revision**, it offers stronger causal interpretability than “overall deception accuracy” in a complex multimodal setting.
- GALILEO likely offers simpler, more reproducible text protocols than video-heavy benchmarks.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not model **multi-party epistemic state** explicitly, it may miss an important driver of conversational failures (ToM / common ground).
- If GALILEO is text-only, it cannot test grounding failures that arise from nonverbal social cues.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a **multi-party** condition (>=3 agents/users) where “who knows what” differs across participants; evaluate whether drift increases with epistemic complexity.
- [ ] Add an analysis section mapping failures to **(i) salience/noise** and **(ii) ToM/epistemic modeling** (even in text-only settings).
- [ ] If GALILEO includes memory/state tracking, frame it as an analogue of **epistemic memory** (who asserted what; what evidence was provided; what each speaker should know).

## Quotes / details to potentially cite

- “state-of-the-art Multimodal Large Language Models (MLLMs) demonstrably lack … the ability to ‘read the room’ and assess deception in complex social interactions.”
- Introduces “Multimodal Interactive Deception Assessment (MIDA)” with “synchronized video and text with verifiable ground-truth labels for every statement.”
- Failure modes: models “fail to effectively ground language in multimodal social cues and lack the ability to model what others know, believe, or intend.”
