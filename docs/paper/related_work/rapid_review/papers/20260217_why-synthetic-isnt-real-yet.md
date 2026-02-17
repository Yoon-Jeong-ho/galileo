# Why Synthetic Isn’t Real Yet: A Diagnostic Framework for Contact Center Dialogue Generation

- Year: 2025
- Venue: arXiv (cs.CL, cs.AI)
- Authors: Rishikesh Devanatha, Varun Nathan, Ayush Kumar
- URL: https://arxiv.org/abs/2508.18210
- BibTeX key (if we add it): why_synthetic_isnt_real_yet_2025
- Tags: synthetic-data, dialogue-generation, diagnostics, evaluation, multi-turn

## One-sentence takeaway

Even with structured supervision, synthetic contact-center dialogues still miss key *multi-turn realism* signals (sentiment arcs, disfluencies, interaction dynamics), and this shows up as a measurable gap on downstream AutoQA—motivating diagnostic, distribution-level evaluation rather than surface fluency metrics.

## What problem does it solve?

- Contact-center transcripts are scarce / sensitive (privacy), so teams want synthetic conversations.
- But “plausible text” is not the same as *realistic* agent–customer interaction; synthetic data may fail downstream.
- The paper aims to (i) benchmark attribute-conditioned generation strategies, and (ii) provide a diagnostic evaluation suite that pinpoints where synthetic dialogues diverge from real ones.

## What is the core method / protocol?

- **Attribute-guided synthetic dialogue generation** conditioned on structured call-center artifacts:
  - intent summaries
  - topic flows
  - QA forms
  - plus target call length + language
- Compare **generation strategies** of increasing structure (as described in the paper):
  - *Direct generation*: single-pass transcript generation conditioned on attributes, with prompt instructions to include call-center characteristics (e.g., disfluency/ASR noise).
  - *Chunked enhancement*: split a base transcript into LLM-chosen chunks and apply characteristic edits per chunk (sampled across dimensions like sentiment / question type), then concatenate.
  - *Characteristic-aware enhancement*: turn-level targeted rewriting where turn features are sampled to match real-data distributions (more controlled than uniform sampling).
- **Downstream utility test**: use synthetic vs real transcripts for **prompt optimization** for an automated QA (AutoQA) task; compare performance.
- **Diagnostic evaluation framework**: 17 metrics across 4 dimensions (their framing):
  1) Emotional & sentiment arcs
  2) Linguistic complexity
  3) Interaction style
  4) Conversational properties (e.g., disfluencies / realism markers)
  
## What are the key metrics?

- AutoQA downstream performance when prompts are optimized on:
  - real transcripts vs synthetic transcripts
- Diagnostic suite (distributional comparisons against real transcripts) spanning:
  - sentiment/emotion arc fidelity
  - disfluency / ASR-noise realism
  - interaction dynamics / behavioral variation
  - conversational structural properties

## What are the main results?

- **Downstream gap**: prompts optimized on **real** transcripts consistently outperform those optimized on **synthetic** transcripts for AutoQA.
- **Even structured supervision is insufficient**: synthetic data shows measurable deficiencies in:
  - sentiment fidelity (affective trajectories)
  - disfluency modeling
  - behavioral variation
  - overall conversational realism
- Main conclusion: evaluation should be **diagnostic and metric-driven**; standard overlap/fluency-style metrics miss the failure modes that matter downstream.

## How is this similar to GALILEO?

- Shares the theme that **multi-turn behavior is multi-dimensional** and needs careful measurement beyond simple accuracy/overlap.
- The “**distribution-level diagnostic**” philosophy aligns with GALILEO’s need to characterize *how* and *when* multi-turn behavior drifts/fails (not just whether it fails).

## How is this different from GALILEO?

- Domain focus: **synthetic contact-center conversations** (goal-oriented spoken dialogue realism), not belief drift / persuasion / sycophancy.
- Evaluates **data generation** and downstream QA utility, rather than robustness to adversarial follow-ups / pressure.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO is (presumably) more directly targeted at multi-turn robustness phenomena of interest (pressure, drift vs revision controls, recovery dynamics).
- This paper’s protocol is domain-specific and realism-focused; it does not provide a clean causal separation of *pressure-driven* vs *evidence-driven* updates.

## Where GALILEO is weaker / needs to improve

- If GALILEO relies on any synthetic dialogue generation (for scaling protocols or interventions), this paper is a warning that **synthetic ≠ real**, and we may need better realism diagnostics or real-data validation.
- GALILEO could benefit from adopting “arc”-style diagnostics (not only end-state correctness).

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a small set of **trajectory-shape diagnostics** (analogous to “sentiment arcs”) for GALILEO dialogues, e.g., confidence/hedging arc, refusal-strength arc, stance/commitment arc.
- [ ] If we generate any synthetic multi-turn dialogues for evaluation/training, include a section arguing **why they are realistic enough**, and/or add diagnostic checks comparing distributions to a real-dialogue reference set.
- [ ] Use this paper as a citation for the claim: “**surface plausibility metrics can miss downstream failures**; diagnostic evaluation is needed for multi-turn interaction quality.”

## Quotes / details to potentially cite

- “We introduce a diagnostic evaluation framework comprising **17 metrics across four dimensions**: (1) Emotional and Sentiment Arcs, (2) Linguistic Complexity, (3) Interaction Style, and (4) Conversational Properties.”
- “Prompts optimized on **real transcripts** consistently outperform those optimized on **synthetic transcripts**” on AutoQA (downstream utility gap).
- Synthetic transcripts show deficiencies in “**sentiment fidelity, disfluency modeling, behavioral variation, and conversational realism**.”
