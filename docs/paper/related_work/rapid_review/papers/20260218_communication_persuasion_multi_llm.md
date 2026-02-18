# Communication is All You Need: Persuasion Dataset Construction via Multi-LLM Communication

- Year: 2025
- Venue: NAACL 2025 (main)
- Authors: Weicheng Ma, Hefan Zhang, Ivory Yang, Shiyu Ji, Joice Chen, Farnoosh Hashemi, Shubham Mohole, Ethan Gearey, Michael Macy, Saeed Hassanpour, Soroush Vosoughi
- URL: https://arxiv.org/abs/2502.08896
- BibTeX key (if we add it): Ma2025CommunicationPersuasion
- Tags: persuasion, synthetic-data, multi-agent, dialogue, evaluation

## One-sentence takeaway

A role-separated multi-LLM “communication” pipeline (generation + monitoring + refinement + global regulation + labeling + postprocessing) produces persuasive dialogues that humans find close to human-written and diverse in strategy use.

## What problem does it solve?

- Persuasion/dialogue datasets are expensive to create/annotate and existing LLM-generated persuasion dialogues can be short, shallow, and unnatural.
- Need scalable generation of longer, coherent persuasive interactions (including ethically challenging/taboo scenarios) with usable per-utterance/over-time persuasion labels.

## What is the core method / protocol?

- A multi-agent framework with ~6 functional agent “groups”:
  - Dialogue generation agents: cyclic turn-taking to produce multi-round conversations.
  - Utterance quality monitor: flags incomplete/repetitive/off-topic turns and requests rewrites; maintains dialogue/topic memory.
  - Language refinement: removes overly polite/boilerplate softeners and makes turns more conversational and argument-focused.
  - Persuasiveness annotation: continuous score in [0,1] tracking cumulative viewpoint shift over rounds (not binary labels).
  - Global regulation: enforces logical flow, reduces repetition/strategy reuse, and decides when to stop the dialogue.
  - Postprocessing: final smoothing/naturalness pass; merges/reassigns labels if structure changes.
- Scenarios seeded from NormBank norms/taboos; supports adversarial settings (both try to persuade) and can be extended to multi-party persuasion with minor changes.
- Implementation detail (as reported): mostly GPT-3.5 backbone; “monitor” and “global regulation” use a stronger model for memory/reasoning.

## What are the key metrics?

- Human “LLM-vs-human” utterance differentiation accuracy on paired (generated vs human-rewritten) sentences.
- Dialogue-level human ratings (Likert 1–3) on coherence, informativeness, overall fluency, role consistency, topic consistency, with inter-rater agreement (Cohen’s kappa / weighted kappa).
- Strategy diversity: human-annotated persuasion strategy categories; similarity of strategy distributions (cosine similarity heatmap).

## What are the main results?

- Utterance naturalness: humans were close to random at identifying which sentence was model-generated vs human-rewritten (both annotators correct in ~29% vs 25% random baseline; substantial disagreement across pairs).
- Dialogue quality ratings are generally high for coherence/clarity/topic/role consistency; weaker for “introducing new information” and overall naturalness (authors attribute to repetition, overly formal tone, informativeness decay in later rounds).
- Strategy diversity: strategy mixes vary across topics and also within-topic across multiple generations; optional “strategy-controlled” prompting works without breaking quality.
- Works in taboo/ethically challenging prompts, but paper discusses risks/misuse and relies on model moderation + observed failures on immoral intents as partial mitigation.

## How is this similar to GALILEO?

- Uses multi-agent role decomposition to improve long-horizon generation quality (monitoring/regulation vs pure single-pass generation).
- Explicitly targets coherence, diversity, and controllability across scenarios—likely overlapping with GALILEO’s goals if GALILEO generates/evaluates complex interactions.

## How is this different from GALILEO?

- Focus is persuasion dialogue dataset construction and labeling (continuous “persuasion shift” scores), not necessarily GALILEO’s domain/task.
- Relies heavily on LLM-as-agent orchestration plus human evaluation; no indication of grounding in external tools/environments (beyond NormBank prompt seeds).
- The quality control is mostly textual (monitor/regulator/refiner) rather than environment-based feedback.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has formal task definitions, grounded evaluation, or stronger safety/constraint handling, it may avoid “LLM polish / repetition” failure modes and provide more objective success metrics.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently uses fewer specialized roles, this paper suggests dedicated “utterance monitor” + “global regulation” roles can materially improve coherence and reduce degenerate loops.
- Continuous, cumulative labeling of state change over interaction could be a useful alternative to binary per-turn labels.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding (or explicitly naming) two governance roles in GALILEO writeup: (1) local-turn quality monitor with memory; (2) global regulator that checks logical influence + repetition and decides termination.
- [ ] If GALILEO needs “change over time” supervision, consider continuous/cumulative scoring rather than binary labels.
- [ ] Add a small human study design idea: “human vs model” discrimination on rewritten pairs as a proxy for utterance naturalness.

## Quotes / details to potentially cite

- “Our framework incorporates 6 groups of language agents…” (multi-agent roles: generation, monitor, refinement, annotation, global regulation, postprocessing).
- Uses a “continuous labeling scheme to measure the degree of perspective change throughout the dialogue, avoiding the limitations of binary utterance labels.”
- Human differentiation results: both annotators correctly pick model-generated in 29.25% of pairs vs 25% random baseline (suggesting near-human utterance quality).
