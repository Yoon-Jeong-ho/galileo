# PingPong: A Natural Benchmark for Multi-Turn Code-Switching Dialogues

- Year: 2026
- Venue: arXiv (cs.CL) preprint
- Authors: Mohammad Farhansyah Rifqi (arXiv submission page lists submitter; full author list not captured from abstract page)
- URL: https://arxiv.org/abs/2601.17277
- BibTeX key (if we add it): rifqi2026pingpong
- Tags: benchmark, multilingual, code-switching, multi-party, multi-turn, dialogue, robustness

## One-sentence takeaway

PingPong is a human-authored, multi-party, multi-turn code-switching dialogue benchmark with tasks (QA, summarization, topic classification) showing current LLMs remain brittle on code-switched inputs.

## What problem does it solve?

- Existing code-switching benchmarks often miss real conversational complexity (multi-party, long-range references, mixed languages, natural structure), limiting their usefulness for evaluating robustness in realistic multilingual settings.

## What is the core method / protocol?

- Collect human-authored conversations with 2–4 participants.
- Cover five language-combination variations, including some trilingual settings.
- Preserve natural, multi-threaded dialogue structure (including long reply distances where a message responds to much earlier turns).
- Define three downstream evaluation tasks derived from the dialogues:
  - Question Answering
  - Dialogue Summarization
  - Topic Classification
- Evaluate multiple state-of-the-art language models on these tasks using code-switched inputs.

## What are the key metrics?

- Standard task performance metrics (not specified on the arXiv abstract page).
- Dataset “naturalness/structure” comparisons vs machine-generated alternatives using descriptive statistics:
  - message length variation
  - speaker dominance variation
  - reply distance (how far back a reply references)

## What are the main results?

- The dataset is reported to be more natural and structurally diverse than machine-generated alternatives.
- Across evaluated SOTA LMs, performance is still limited on code-switched inputs, motivating more robust multilingual NLP.

## How is this similar to GALILEO?

- Shares the general theme of *robustness evaluation*: stress-testing models under distributional / interactional complexity (here: multilingual code-switching + multi-party + long-context dependencies).
- Highlights failure modes that may emerge in multi-turn settings with long-range references.

## How is this different from GALILEO?

- Focuses on multilingual dialogue and code-switching rather than GALILEO’s specific instability/drift/susceptibility protocols.
- Tasks are downstream NLP tasks (QA/summarization/topic classification) rather than targeted behavioral robustness measures (e.g., flip rates, time-to-failure, recovery).

## Where GALILEO is stronger / cleaner (if true)

- GALILEO-style protocols can isolate specific robustness mechanisms (drift vs evidence, recovery, intervention) more directly than downstream task scores.

## Where GALILEO is weaker / needs to improve

- If GALILEO claims broad conversational robustness, it may need multilingual and code-switching coverage (multi-party + long-range reference) as additional stressors.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a “code-switching / multilingual” stressor dimension (at least a small pilot) to test whether instability metrics degrade under language mixing.
- [ ] In related work, position PingPong as a *naturalistic multi-turn benchmark* and note that even strong models struggle when dialogue structure becomes realistic (multi-party, long reply distance).

## Quotes / details to potentially cite

- “We present PingPong, a benchmark for natural multi-party code-switching dialogues … human-authored conversations among 2 to 4 participants … replies frequently reference much earlier points in the dialogue.” (arXiv abstract)
- “Based on these dialogues, we define three downstream tasks: Question Answering, Dialogue Summarization, and Topic Classification.” (arXiv abstract)
- “Evaluations … reveal that performance remains limited on code-switched inputs … need for more robust NLP systems … real-world multilingual discourse.” (arXiv abstract)
