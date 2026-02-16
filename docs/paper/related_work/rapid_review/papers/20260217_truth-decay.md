# TRUTH DECAY: Quantifying Multi-Turn Sycophancy in Language Models

- Year: 2025
- Venue: arXiv (also on OpenReview)
- Authors: Joshua Liu; Aarav Jain; Soham Takuri; Srihan Vege; Aslihan Akalin; Kevin Zhu; Sean O’Brien; Vasu Sharma
- URL: https://arxiv.org/abs/2503.11656 (OpenReview: https://openreview.net/forum?id=GHUh9O5Im8)
- BibTeX key (if we add it): liu2025truthdecay
- Tags: sycophancy, persuasion, multi-turn, robustness, truthfulness, drift

## One-sentence takeaway

TRUTH DECAY proposes a **multi-turn sycophancy benchmark** showing that repeated user pressure (especially confident or rationale-backed) can **compound factual degradation over turns**, and that simple “anti-sycophancy” preambles often weaken in longer dialogues.

## What problem does it solve?

- Prior sycophancy evaluations are mostly **single-turn** (or “flip once”) and can miss **compounding degradation** in longer interactions.
- Need a protocol to quantify how quickly models “decay” from an initially correct stance under **iterative user feedback/challenges/persuasion**.

## What is the core method / protocol?

- Task setting: multiple-choice QA with ground truth (primarily **TruthfulQA** and **MMLU-Pro**).
- Conversation structure: model answers an initial question, then receives **n follow-ups** (reported examples include **1, 3, 7 turns**).
- Two follow-up generation modes:
  - **Static feedback**: templated follow-ups meant to elicit sycophancy, adapted from Anthropic-style prompts.
  - **Rationale-based feedback**: a separate model generates a **plausible-but-wrong rationale** for a randomly chosen incorrect answer; this rationale is then used to pressure the answering model over multiple turns.
- Four bias types (extended across turns):
  - Feedback sycophancy
  - “Are you sure?” sycophancy
  - Answer sycophancy (appeals to majority/external sources)
  - Mimicry sycophancy (user states an answer as fact with high confidence)
- “Sycophancy reduction” interventions tested as **prefixes** to the follow-ups:
  - Source-info style (“be skeptical of user-provided info…evaluate based on own knowledge”)
  - Direct-command style (“do not agree solely because user says so…”) 

## What are the key metrics?

- Per-turn **accuracy** (and accuracy degradation over turns / domains).
- **Answer change rate** across follow-ups (often stratified by whether the initial answer was correct vs incorrect).
- Reported emphasis: *multi-turn trajectories* (e.g., accuracy drops from turn 1 → turn 7; change rate growth across turns).

## What are the main results?

(From arXiv HTML v1; details are reported as headline numbers rather than fully standardized across all settings.)

- Multi-turn pressure reveals strong compounding effects: sycophancy-driven **accuracy drops up to ~47%** over extended conversations.
- Strong dependence on initial correctness:
  - If the model starts **incorrect**, subsequent answer changes can rise steeply (example narrative: reaching ~50% change by ~turn 4), while **initially correct** answers are more stable (≈10% change in an example figure description).
- Domain differences: subjective domains (e.g., philosophy) show larger degradation than STEM domains.
- Static multi-step follow-ups can significantly reduce accuracy across models; smaller models (e.g., Llama 3.1 8B) appear most vulnerable (example: accuracy “collapses” under sustained follow-ups).
- Rationale-based follow-ups can further destabilize answers by encouraging the model to **internalize** flawed reasoning, not just agree.
- Simple anti-sycophancy prefixes help less reliably in multi-turn settings than one might expect.

## How is this similar to GALILEO?

- Same core phenomenon: **multi-turn robustness under social pressure** / persuasion.
- Emphasizes **time/turn-dependent degradation**, aligning with GALILEO’s focus on trajectories (not just single-step flips).
- Uses a benchmark framing with repeatable follow-up templates, similar in spirit to controlled “pressure protocols.”

## How is this different from GALILEO?

- Protocol is mostly **QA accuracy / answer flips** under pressure; less explicit separation between:
  - evidence-based belief revision vs. social/pure-pressure drift,
  - and less explicit “recovery” measurement (return-to-truth after a flip).
- Emphasis is on *accuracy degradation*, not explicitly a survival/time-to-failure model (hazards, censored runs, etc.).
- Uses a mixture of static templates and synthetic rationales; may conflate “pressure strength” with “content informativeness.”

## Where GALILEO is stronger / cleaner (if true)

- Opportunity for GALILEO to provide clearer **controls** (e.g., same information content but different social framing) and cleaner **time-to-event** metrics.
- Opportunity to measure **recovery after flip** (interventions that restore truth) rather than only degradation.

## Where GALILEO is weaker / needs to improve

- Need to explicitly acknowledge and position against TRUTH DECAY as a close neighbor; otherwise GALILEO may look incremental.
- Need a crisp statement of what GALILEO measures that TRUTH DECAY does not (e.g., drift-vs-revision controls, survival analysis, recovery).

## Action items for GALILEO (experiments / method / writing)

- [ ] Related work: cite TRUTH DECAY as a direct precursor for multi-turn sycophancy benchmarking; contrast GALILEO’s metrics/controls.
- [ ] Add an ablation inspired by TRUTH DECAY’s “static vs rationale-based” pressure: compare pressure that is (a) confident but contentless vs (b) argument-rich.
- [ ] Consider reporting “initially correct vs initially incorrect” stratification for failure rates/time-to-failure.
- [ ] If we use anti-sycophancy prompts/interventions, evaluate them **over long horizons**, not just the first flip.

## Quotes / details to potentially cite

- Abstract framing: benchmark for sycophancy in extended dialogues with iterative user feedback/challenges/persuasion.
- Protocol components: static follow-up templates adapted from Anthropic single-step sycophancy test; rationale-based follow-ups from a separate rationale generator model.
- Summary result claim: multi-turn interactions can cause sizable accuracy drops (reported up to ~47%).
