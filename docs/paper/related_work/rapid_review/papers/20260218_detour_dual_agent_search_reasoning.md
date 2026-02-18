# DETOUR: An Interactive Benchmark for Dual-Agent Search and Reasoning

- Year: 2026
- Venue: arXiv
- Authors: Siyan Li; Darshan Deshpande; Anand Kannappan; Rebecca Qian
- URL: https://arxiv.org/abs/2602.00352
- BibTeX key (if we add it): detour2026
- Tags: agents, search, reasoning, interactive, multi-turn, benchmark

## One-sentence takeaway

DETOUR is a 1,011-item, multi-turn dual-agent benchmark for “tip-of-the-tongue” known-item search that stresses *clarification-seeking under underspecification*, where even strong models degrade sharply in multimodal settings.

## What problem does it solve?

- Existing agent/search benchmarks largely evaluate **single-turn** queries, missing the realistic phenomenon that people recall targets after **multiple clarification turns**.
- Prior work on ambiguous/uncertain search often measures end accuracy but not the **quality/usefulness of follow-up questions** in an interactive loop.

## What is the core method / protocol?

- **Two-agent evaluation**:
  - **Primary Agent** (the system under test) must identify the target entity.
  - **Memory Agent** is fixed/held constant and can be queried; it has access to a “memory file” with extra cues that *do not directly reveal* the answer.
- Primary Agent can iterate over turns, asking clarification questions (to the Memory Agent) and using web search, aiming to resolve an underspecified recollection.
- Benchmark size and scope: **1,011 prompts**, spanning multiple domains and **modalities** (text, image, audio, video).

## What are the key metrics?

- **Accuracy** (reported for different modality splits; key headline numbers are text-only vs all-modality).
- (From the paper framing) qualitative failure analysis around whether follow-up queries actually reduce ambiguity.

## What are the main results?

- Reported in the paper:
  - ~**66%** accuracy on the **text-only** split for a strong frontier model.
  - Drops to ~**36%** accuracy when evaluated on **all modalities** together (text+image+audio+video).
- Noted failure mode: agents often **query the Memory Agent repeatedly “in vain”**, derailing reasoning instead of asking incisive disambiguating questions.

## How is this similar to GALILEO?

- Shared emphasis on **multi-turn evaluation** rather than single-turn snapshots.
- Treats interactive behavior (question-asking / tool-using) as central, not just final answers.
- Highlights that **underspecification/ambiguity** is a key driver of model failure—useful context for GALILEO’s claims about brittle multi-turn dynamics.

## How is this different from GALILEO?

- DETOUR focuses on **known-item retrieval / search** with a Memory Agent + web search loop, not on social pressure / belief drift vs evidence-driven revision.
- Core outcome is **retrieval identification accuracy**, rather than stability/recovery metrics (e.g., time-to-flip, recovery-after-flip) under controlled pressure operators.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO explicitly separates **pressure-only drift** from **evidence-based updating** with matched controls, that is a cleaner causal story than DETOUR’s broad interactive setting.
- GALILEO can likely offer more **trajectory-aware metrics** (flip/recovery structure) beyond end accuracy.

## Where GALILEO is weaker / needs to improve

- DETOUR’s design pressure-tests **clarification question quality** in a grounded, interactive loop; GALILEO may need a clearer story/metric for “did the agent ask the *right* follow-ups?” when ambiguity is present.
- DETOUR’s multi-modality stressor (image/audio/video) suggests a gap if GALILEO is primarily text.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a small “underspecified query” slice where success requires **asking discriminative clarification questions** (even in purely text form), and score question usefulness.
- [ ] Consider a **two-role evaluation harness** (evaluated agent + fixed responder/“memory” oracle) to standardize multi-turn interactions and reduce evaluator variance.
- [ ] In writing: cite DETOUR as evidence that **interactive ambiguity resolution remains hard**, and argue why GALILEO’s controlled operators/metrics complement this benchmark.

## Quotes / details to potentially cite

- “... we introduce ... (DETOUR), a dual-agent evaluation benchmark containing **1,011 prompts**.”
- “... only achieving **36% accuracy** when evaluated on all modalities (text, image, audio, and video) ...”
- “... GPT-5 achieve only **66% accuracy** on the text-only split ...”
